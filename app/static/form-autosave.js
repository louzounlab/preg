(function () {
    'use strict';

    var PREFIX = 'autosave:';

    function storageKey(form) {
        return PREFIX + form.dataset.autosaveKey;
    }

    function snapshot(form) {
        var data = {};
        form.querySelectorAll('input, select, textarea').forEach(function (el) {
            if (!el.name) return;
            var t = el.type;
            if (t === 'submit' || t === 'button' || t === 'file' || t === 'reset') return;
            if (t === 'checkbox' || t === 'radio') {
                data[el.name] = el.checked;
            } else {
                data[el.name] = el.value;
            }
        });
        return data;
    }

    function restore(form, data) {
        Object.keys(data).forEach(function (name) {
            var el = form.querySelector('[name="' + name + '"]');
            if (!el) return;
            var t = el.type;
            if (t === 'checkbox' || t === 'radio') {
                if (!el.checked && data[name]) el.checked = true;
            } else if (el.value === '' || el.value === 'None') {
                el.value = data[name];
                if (el.value !== '' && el.value !== 'None' && el.hidden) {
                    el.hidden = false;
                }
            }
        });
    }

    function save(form) {
        try {
            localStorage.setItem(storageKey(form), JSON.stringify(snapshot(form)));
        } catch (e) { /* storage full or disabled */ }
    }

    function load(form) {
        try {
            var raw = localStorage.getItem(storageKey(form));
            return raw ? JSON.parse(raw) : null;
        } catch (e) { return null; }
    }

    function clearStored(form) {
        try { localStorage.removeItem(storageKey(form)); } catch (e) { /* ignore */ }
    }

    // Twin-PE: pre-create Second/Third trimester frames so their inputs exist before restore.
    function preparePEFrames(form, data) {
        if (typeof window.addTrimesterFrame !== 'function') return;
        var keys = Object.keys(data);
        if (keys.some(function (k) { return k.indexOf('Trim_2_') === 0; }) &&
            !form.querySelector('.second-trimester')) {
            window.addTrimesterFrame();
        }
        if (keys.some(function (k) { return k.indexOf('Trim_3_') === 0; }) &&
            !form.querySelector('.third-trimester')) {
            window.addTrimesterFrame();
        }
    }

    // Twin-EFW: unhide rows 5..last_row after restore so populated cells are visible.
    function expandEFWRows(form) {
        var lastRow = form.querySelector('[name="last_row"]');
        if (!lastRow) return;
        var lr = parseInt(lastRow.value, 10);
        if (isNaN(lr)) return;
        ['week', 'Day', 'EFW1_', 'EFW2_'].forEach(function (prefix) {
            for (var i = 5; i <= Math.min(lr, 10); i++) {
                var el = form.querySelector('#' + prefix + i);
                if (el) el.hidden = false;
            }
        });
    }

    function attach(form) {
        var data = load(form);
        if (data) {
            preparePEFrames(form, data);
            restore(form, data);
            expandEFWRows(form);
        }

        var persist = function () { save(form); };
        form.addEventListener('input', persist);
        form.addEventListener('change', persist);
        // Buttons (add row, MCDA/DCDA, add/remove trimester) mutate state without firing input/change.
        form.addEventListener('click', function (e) {
            if (e.target.closest('button')) setTimeout(persist, 0);
        });

        // Sync current (possibly server-rendered) state into storage as the new baseline.
        save(form);

        var clearBtn = form.querySelector('[data-autosave-clear]');
        if (clearBtn) {
            clearBtn.addEventListener('click', function (e) {
                e.preventDefault();
                if (!window.confirm('Clear all saved inputs for this form?')) return;
                clearStored(form);
                window.location.href = window.location.pathname;
            });
        }
    }

    function init() {
        document.querySelectorAll('form[data-autosave-key]').forEach(attach);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
