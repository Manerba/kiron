"""Node-basierte Vertragstests fuer die 1m-/10m-Logik im Verlauf-Tab."""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


NODE = shutil.which("node")
SOURCE = Path(__file__).resolve().parent / "static/js/tab_history.js"
CHART_SOURCE = Path(__file__).resolve().parent / "static/js/chart_component.js"


@unittest.skipUnless(NODE, "node ist fuer Frontend-Vertragstest nicht installiert")
class HistoryLiveFrontendTests(unittest.TestCase):
    def test_live_charts_use_static_relative_ticks(self):
        harness = r"""
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[1], 'utf8');

class FakeChart {
    constructor(_context, config) {
        this.data = config.data;
        this.options = config.options;
        this.updateModes = [];
    }
    update(mode) { this.updateModes.push(mode); }
    destroy() {}
}

const context = {
    Chart: FakeChart,
    Date,
    Number,
    document: {
        documentElement: {},
        getElementById: () => ({ getContext: () => ({}) }),
    },
    getComputedStyle: () => ({ getPropertyValue: () => '' }),
};
vm.createContext(context);
vm.runInContext(source + '\nglobalThis.__TimeSeriesChart = TimeSeriesChart;', context);

const chart = new context.__TimeSeriesChart('chart', {
    series: [{ key: 'cpu_usage', label: 'CPU', color: '#123456', yAxisID: 'y' }],
    yAxes: [{ id: 'y', label: '%', min: 0, max: 100 }],
});

const firstTimestamps = [100, 110, 120, 130, 140, 150, 160];
chart.update(firstTimestamps, { cpu_usage: [1, 2, 3, 4, 5, 6, 7] }, {
    relativeWindowSeconds: 60,
});
let xAxis = chart.chart.options.scales.x;
assert.strictEqual(xAxis.type, 'linear');
assert.strictEqual(xAxis.min, 0);
assert.strictEqual(xAxis.max, 60);
assert.strictEqual(xAxis.reverse, true);
assert.strictEqual(xAxis.ticks.stepSize, 10);
assert.strictEqual(xAxis.ticks.autoSkip, false);
assert.deepStrictEqual(
    [0, 10, 20, 30, 40, 50, 60].map(xAxis.ticks.callback),
    ['0 s', '10 s', '20 s', '30 s', '40 s', '50 s', '60 s'],
);
assert.deepStrictEqual(
    Array.from(chart.chart.data.datasets[0].data, point => point.x),
    [60, 50, 40, 30, 20, 10, 0],
);

chart.update(firstTimestamps.map(value => value + 0.25), {
    cpu_usage: [1, 2, 3, 4, 5, 6, 7],
}, { relativeWindowSeconds: 60 });
assert.deepStrictEqual(
    Array.from(chart.chart.data.datasets[0].data, point => point.x),
    [60, 50, 40, 30, 20, 10, 0],
    'relative labels and positions must not drift with wall-clock time',
);

const tenMinuteTimestamps = [100, 220, 340, 460, 580, 700];
chart.update(tenMinuteTimestamps, { cpu_usage: [1, 2, 3, 4, 5, 6] }, {
    relativeWindowSeconds: 600,
});
xAxis = chart.chart.options.scales.x;
assert.strictEqual(xAxis.type, 'linear');
assert.strictEqual(xAxis.min, 0);
assert.strictEqual(xAxis.max, 600);
assert.strictEqual(xAxis.reverse, true);
assert.strictEqual(xAxis.ticks.stepSize, 120);
assert.deepStrictEqual(
    [0, 120, 240, 360, 480, 600].map(xAxis.ticks.callback),
    ['0 min', '2 min', '4 min', '6 min', '8 min', '10 min'],
);
assert.deepStrictEqual(
    Array.from(chart.chart.data.datasets[0].data, point => point.x),
    [600, 480, 360, 240, 120, 0],
);

chart.update([170], { cpu_usage: [8] });
xAxis = chart.chart.options.scales.x;
assert.strictEqual(xAxis.type, 'time');
assert.ok(chart.chart.data.datasets[0].data[0].x instanceof Date);
"""
        completed = subprocess.run(
            [NODE, "-e", harness, str(CHART_SOURCE)],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"node stdout:\n{completed.stdout}\nnode stderr:\n{completed.stderr}",
        )

    def test_range_buffer_stream_and_cleanup_contract(self):
        harness = r"""
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[1], 'utf8');

const sockets = [];
const animationFrames = new Map();
const timers = new Map();
let nextHandle = 1;
let fetchCalls = [];

class FakeSocket {
    constructor(url) {
        this.url = url;
        this.closed = false;
        sockets.push(this);
    }
    close(code, reason) {
        this.closed = true;
        this.closeCode = code;
        this.closeReason = reason;
    }
}

function fakeNode() {
    return {
        children: [],
        className: '',
        textContent: '',
        appendChild(child) { this.children.push(child); },
        replaceChildren() { this.children = []; },
    };
}

const statsNode = fakeNode();
const context = {
    console,
    Map,
    Set,
    Date,
    JSON,
    Number,
    Object,
    Array,
    Math,
    currentTab: 'history',
    WebSocket: FakeSocket,
    fetch: url => {
        fetchCalls.push(url);
        return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
    },
    setInterval: () => nextHandle++,
    clearInterval: () => {},
    setTimeout: callback => {
        const handle = nextHandle++;
        timers.set(handle, callback);
        return handle;
    },
    clearTimeout: handle => timers.delete(handle),
    document: {
        getElementById: id => id === 'historyStats' ? statsNode : null,
        querySelectorAll: () => [],
        createElement: () => fakeNode(),
        createTextNode: text => ({ textContent: text }),
    },
    window: {
        location: { protocol: 'https:', host: 'kiron.test' },
        requestAnimationFrame: callback => {
            const handle = nextHandle++;
            animationFrames.set(handle, callback);
            return handle;
        },
        cancelAnimationFrame: handle => animationFrames.delete(handle),
    },
};
vm.createContext(context);
vm.runInContext(source + `
globalThis.__historyTest = {
    ranges: () => HISTORY_RANGES,
    activeRange: () => historyActiveRange,
    setRange: value => { historyActiveRange = value; },
    switchRange: setHistoryRange,
    replaceSamples: replaceHistoryLiveSamples,
    appendSample: appendHistoryLiveSample,
    samples: () => historyLiveSamples,
    stop: stopHistoryLive,
};
`, context);

const test = context.__historyTest;
assert.strictEqual(test.activeRange(), '1m');
assert.strictEqual(test.ranges()['1m'].seconds, 60);
assert.strictEqual(test.ranges()['1m'].live, true);
assert.strictEqual(test.ranges()['1m'].refreshMs, 0);
assert.strictEqual(test.ranges()['1m'].maxPoints, 256);
assert.strictEqual(test.ranges()['10m'].seconds, 600);
assert.strictEqual(test.ranges()['10m'].live, true);
assert.strictEqual(test.ranges()['10m'].refreshMs, 0);
assert.strictEqual(test.ranges()['10m'].maxPoints, 2416);

const makeSample = (sequence, timestamp, cpu = sequence) => ({
    sequence,
    timestamp,
    values: {
        cpu_usage: cpu,
        cpu_load_1m: 1,
        memory_usage: 2,
        memory_used_gb: 3,
        gpu_util: 4,
        vram_usage: 5,
        disk_read_mb_s: 6,
        disk_write_mb_s: 7,
    },
    states: { cpu: 'ok', memory: 'ok', disk: 'ok', gpu: 'ok' },
});

const oversized = [];
for (let index = 0; index < 300; index += 1) {
    oversized.push(makeSample(index + 1, 1000 + index * 0.25));
}
test.replaceSamples(oversized.reverse());
let samples = test.samples();
assert.ok(samples.length <= 256, `unbounded sample count: ${samples.length}`);
assert.ok(samples[0].timestamp > samples[samples.length - 1].timestamp - 60);
assert.strictEqual(
    samples.map(sample => sample.sequence).join(','),
    [...samples].map(sample => sample.sequence).sort((a, b) => a - b).join(','),
);
const latestSequence = samples[samples.length - 1].sequence;
assert.strictEqual(test.appendSample(makeSample(latestSequence, 2000)), false);
assert.strictEqual(test.appendSample(makeSample(latestSequence + 1, 2000.25)), true);

test.setRange('1h');
test.switchRange('1m');
assert.strictEqual(fetchCalls.length, 0, '1m must not call the DB history API');
assert.strictEqual(sockets.length, 1);
assert.strictEqual(sockets[0].url, 'wss://kiron.test/ws/history/1m');

sockets[0].onmessage({ data: JSON.stringify({
    type: 'history_1m_snapshot',
    version: 1,
    interval_ms: 250,
    samples: [makeSample(2, 2), makeSample(1, 1), makeSample(2, 2, 22)],
    status: { running: true },
}) });
for (const [handle, callback] of [...animationFrames]) {
    animationFrames.delete(handle);
    callback();
}
samples = test.samples();
assert.strictEqual(samples.map(sample => sample.sequence).join(','), '1,2');
assert.strictEqual(samples[1].values.cpu_usage, 22);

test.switchRange('1h');
assert.strictEqual(sockets[0].closed, true);
assert.strictEqual(sockets[0].closeCode, 1000);
assert.ok(fetchCalls.some(url => url.startsWith('/api/history/metrics?')));
assert.ok(fetchCalls.includes('/api/history/stats'));

const fetchCountBeforeTenMinutes = fetchCalls.length;
test.switchRange('10m');
assert.strictEqual(sockets.length, 2);
assert.strictEqual(sockets[1].url, 'wss://kiron.test/ws/history/10m');
assert.strictEqual(fetchCalls.length, fetchCountBeforeTenMinutes, '10m must not call the DB API');

const tenMinuteSamples = [];
for (let index = 0; index < 2500; index += 1) {
    tenMinuteSamples.push(makeSample(index + 1, 1000 + index * 0.25));
}
sockets[1].onmessage({ data: JSON.stringify({
    type: 'history_10m_snapshot',
    version: 1,
    interval_ms: 250,
    samples: tenMinuteSamples.reverse(),
    status: { running: true },
}) });
samples = test.samples();
assert.ok(samples.length <= 2416, `unbounded 10m sample count: ${samples.length}`);
assert.ok(samples[0].timestamp > samples[samples.length - 1].timestamp - 600);
assert.strictEqual(samples.length, 2400);

test.switchRange('1h');
assert.strictEqual(sockets[1].closed, true);
test.switchRange('10m');
assert.strictEqual(sockets.length, 3);
sockets[2].onclose({ code: 1006 });
assert.strictEqual(timers.size, 1, 'disconnect must schedule exactly one reconnect');
test.switchRange('1h');
assert.strictEqual(timers.size, 0, 'range change must cancel the reconnect');
"""
        completed = subprocess.run(
            [NODE, "-e", harness, str(SOURCE)],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"node stdout:\n{completed.stdout}\nnode stderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
