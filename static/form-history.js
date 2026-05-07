(function () {
    'use strict';

    var HIST_PREFIX = 'formhist:';
    var MAX_PER_FIELD = 10;

    function formKey(form) {
        return form.dataset.autosaveKey || form.getAttribute('action') || form.id || location.pathname;
    }

    function fieldKey(form, name) {
        return HIST_PREFIX + formKey(form) + ':' + name;
    }

    function loadHistory(form, name) {
        try {
            var raw = localStorage.getItem(fieldKey(form, name));
            return raw ? JSON.parse(raw) : [];
        } catch (e) { return []; }
    }

    function saveHistory(form, name, list) {
        try {
            localStorage.setItem(fieldKey(form, name), JSON.stringify(list.slice(0, MAX_PER_FIELD)));
        } catch (e) { /* storage full or disabled */ }
    }

    function addToHistory(form, name, value) {
        if (value === '' || value == null) return;
        var list = loadHistory(form, name).filter(function (v) { return v !== value; });
        list.unshift(value);
        saveHistory(form, name, list);
    }

    function eligible(el) {
        if (!el || (el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA')) return false;
        if (!el.name) return false;
        var t = (el.type || 'text').toLowerCase();
        var skip = ['hidden', 'file', 'password', 'submit', 'button', 'reset',
                    'checkbox', 'radio', 'image', 'color', 'range'];
        return skip.indexOf(t) === -1;
    }

    var current = null; // { form, input, dropdown, items, selected }

    function closeDropdown() {
        if (!current) return;
        if (current.dropdown && current.dropdown.parentNode) {
            current.dropdown.parentNode.removeChild(current.dropdown);
        }
        current = null;
    }

    function positionDropdown(input, dd) {
        var r = input.getBoundingClientRect();
        dd.style.left = (r.left + window.scrollX) + 'px';
        dd.style.top = (r.bottom + window.scrollY) + 'px';
        dd.style.width = r.width + 'px';
    }

    function highlight(idx) {
        if (!current) return;
        Array.prototype.forEach.call(current.dropdown.children, function (row, i) {
            row.style.background = (i === idx) ? '#e6f5f3' : '#fff';
        });
        current.selected = idx;
    }

    function pick(idx) {
        if (!current) return;
        var val = current.items[idx];
        var input = current.input;
        input.value = val;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        closeDropdown();
        input.focus();
    }

    function buildDropdown(form, input) {
        var history = loadHistory(form, input.name);
        var query = (input.value || '').toLowerCase();
        var items = history.filter(function (v) {
            return v !== input.value && (query === '' || String(v).toLowerCase().indexOf(query) !== -1);
        });
        if (items.length === 0) { closeDropdown(); return; }

        var dd;
        if (current && current.input === input) {
            dd = current.dropdown;
            dd.innerHTML = '';
            current.items = items;
            current.selected = -1;
        } else {
            closeDropdown();
            dd = document.createElement('div');
            dd.className = 'form-history-dropdown';
            Object.assign(dd.style, {
                position: 'absolute',
                background: '#fff',
                border: '1px solid #ccc',
                borderTop: 'none',
                borderRadius: '0 0 4px 4px',
                maxHeight: '200px',
                overflowY: 'auto',
                boxShadow: '0 4px 8px rgba(0,0,0,0.15)',
                fontFamily: 'inherit',
                fontSize: '14px',
                color: '#333',
                zIndex: '10000'
            });
            document.body.appendChild(dd);
            current = { form: form, input: input, dropdown: dd, items: items, selected: -1 };
        }

        items.forEach(function (val, idx) {
            var row = document.createElement('div');
            row.textContent = val;
            Object.assign(row.style, { padding: '6px 10px', cursor: 'pointer', background: '#fff' });
            row.addEventListener('mouseenter', function () { highlight(idx); });
            row.addEventListener('mousedown', function (e) {
                e.preventDefault();
                pick(idx);
            });

            var del = document.createElement('span');
            del.textContent = '×';
            Object.assign(del.style, {
                float: 'right', marginLeft: '8px', color: '#999',
                cursor: 'pointer', fontWeight: 'bold'
            });
            del.title = 'Remove this entry';
            del.addEventListener('mousedown', function (e) {
                e.preventDefault();
                e.stopPropagation();
                var list = loadHistory(form, input.name).filter(function (v) { return v !== val; });
                saveHistory(form, input.name, list);
                buildDropdown(form, input);
            });
            row.appendChild(del);

            dd.appendChild(row);
        });

        positionDropdown(input, dd);
    }

    function attach(form) {
        form.addEventListener('focusin', function (e) {
            if (eligible(e.target)) buildDropdown(form, e.target);
        });
        form.addEventListener('input', function (e) {
            if (eligible(e.target)) buildDropdown(form, e.target);
        });
        form.addEventListener('focusout', function (e) {
            setTimeout(function () {
                if (current && current.input === e.target && document.activeElement !== e.target) {
                    closeDropdown();
                }
            }, 150);
        });
        // Capture-phase keydown so we can intercept arrow keys before
        // page-specific handlers (e.g. twin-fwe grid navigation).
        form.addEventListener('keydown', function (e) {
            if (!current || current.input !== e.target) return;
            var n = current.items.length;
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                e.stopImmediatePropagation();
                highlight((current.selected + 1) % n);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                e.stopImmediatePropagation();
                highlight(current.selected <= 0 ? n - 1 : current.selected - 1);
            } else if (e.key === 'Enter' && current.selected >= 0) {
                e.preventDefault();
                e.stopImmediatePropagation();
                pick(current.selected);
            } else if (e.key === 'Escape') {
                closeDropdown();
            } else if (e.key === 'Tab') {
                closeDropdown();
            }
        }, true);
        form.addEventListener('submit', function () {
            form.querySelectorAll('input, textarea').forEach(function (el) {
                if (eligible(el)) addToHistory(form, el.name, el.value);
            });
        });
    }

    function init() {
        document.querySelectorAll('form').forEach(attach);

        var reposition = function () { if (current) positionDropdown(current.input, current.dropdown); };
        window.addEventListener('scroll', reposition, true);
        window.addEventListener('resize', reposition);

        document.addEventListener('mousedown', function (e) {
            if (!current) return;
            if (current.dropdown.contains(e.target) || current.input === e.target) return;
            closeDropdown();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
