/**
 * Shared Frontend Core — SharedUi Namespace
 * Quelle: kiara/web/static/ui_core/js/core.js
 *
 * Erzeugt window.SharedUi idempotent. Kein Produkt-/API-String.
 */
(function () {
    if (window.SharedUi && window.SharedUi.version) {
        return;
    }

    var _config = {
        productName: 'Product',
        themeStorageKey: 'ui_theme',
        allowedThemes: ['light', 'dark', 'kontrast']
    };

    function configure(options) {
        if (!options || typeof options !== 'object') {
            return;
        }
        Object.assign(_config, options);
    }

    window.SharedUi = {
        version: '0.1.0',
        configure: configure,
        _config: function () { return _config; }
    };
})();
