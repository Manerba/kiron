/**
 * Shared Frontend Core — Theme Runtime
 * Quelle: kiara/web/static/ui_core/js/theme.js
 *
 * SharedUi.theme API: load/set/toggle/isAllowed/getAllowed.
 * Benoetigt zuvor geladenes ui_core/js/core.js.
 */
(function () {
    if (!window.SharedUi || typeof window.SharedUi.configure !== 'function') {
        throw new Error('ui_core/core.js must load before theme.js');
    }

    if (window.SharedUi.theme) {
        return;
    }

    var DEFAULT_ALLOWED = ['light', 'dark', 'kontrast'];

    function _normalizedAllowed() {
        var cfg = window.SharedUi._config();
        var raw = cfg && cfg.allowedThemes;
        if (!Array.isArray(raw) || raw.length === 0) {
            return DEFAULT_ALLOWED.slice();
        }
        var out = [];
        for (var i = 0; i < raw.length; i++) {
            if (typeof raw[i] === 'string' && raw[i].length > 0) {
                out.push(raw[i]);
            }
        }
        if (out.length === 0) {
            return DEFAULT_ALLOWED.slice();
        }
        return out;
    }

    function _storageKey() {
        var cfg = window.SharedUi._config();
        return (cfg && cfg.themeStorageKey) || 'ui_theme';
    }

    function _syncSelectors(theme) {
        var nodes = document.querySelectorAll('[data-theme-selector]');
        for (var i = 0; i < nodes.length; i++) {
            var el = nodes[i];
            if ('value' in el) {
                el.value = theme;
            }
        }
    }

    function isAllowed(theme) {
        var allowed = _normalizedAllowed();
        return allowed.indexOf(theme) !== -1;
    }

    function getAllowed() {
        return _normalizedAllowed().slice();
    }

    function load() {
        var allowed = _normalizedAllowed();
        var fallback = allowed[0];
        var stored = null;
        try {
            stored = localStorage.getItem(_storageKey());
        } catch (err) {
            stored = null;
        }
        var applied = (stored && allowed.indexOf(stored) !== -1) ? stored : fallback;
        document.documentElement.setAttribute('data-theme', applied);
        return applied;
    }

    function set(theme) {
        var allowed = _normalizedAllowed();
        if (allowed.indexOf(theme) === -1) {
            return false;
        }
        try {
            localStorage.setItem(_storageKey(), theme);
        } catch (err) {
            return false;
        }
        document.documentElement.setAttribute('data-theme', theme);
        _syncSelectors(theme);
        document.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme: theme } }));
        return true;
    }

    function toggle() {
        var allowed = _normalizedAllowed();
        var current = document.documentElement.getAttribute('data-theme');
        var idx = allowed.indexOf(current);
        var nextIdx = (idx === -1) ? 0 : (idx + 1) % allowed.length;
        return set(allowed[nextIdx]);
    }

    window.SharedUi.theme = {
        load: load,
        set: set,
        toggle: toggle,
        isAllowed: isAllowed,
        getAllowed: getAllowed
    };
})();
