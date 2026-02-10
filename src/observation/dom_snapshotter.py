from playwright.sync_api import Page
from typing import List, Dict

class DOMSnapshotter:
    def __init__(self, page: Page):
        self.page = page

    def capture(self) -> str:
        """Captures a structured DOM snapshot of interactive/visible elements + visible page text."""
        try:
            elements = self._extract_elements()
            formatted = self._format_elements(elements)
            page_text = self._extract_page_text()
            if not page_text:
                # Page may be mid-navigation; brief wait and retry
                self.page.wait_for_timeout(1500)
                page_text = self._extract_page_text()
            if page_text:
                formatted += f"\n\nPAGE TEXT (visible content):\n{page_text}"
            return formatted
        except Exception:
            return ""

    def _extract_page_text(self, max_chars: int = 600) -> str:
        """Extract visible text from the page using innerText (reliable, excludes scripts/styles)."""
        try:
            text = self.page.evaluate("() => (document.body.innerText || '').slice(0, 3000)")
            if not text or len(text.strip()) < 10:
                return ""
            # Collapse whitespace and truncate
            cleaned = " ".join(text.split())
            return cleaned[:max_chars]
        except Exception:
            return ""

    def _extract_elements(self) -> List[Dict]:
        """Extracts interactive elements; typeable inputs first so search box gets e0 when possible."""
        return self.page.evaluate("""() => {
            const INTERACTIVE = 'a, button, input, select, textarea, [role="button"], [role="link"], [role="menuitem"], [role="tab"], [role="checkbox"], [role="radio"], [role="textbox"], [role="combobox"], [role="option"]';
            const all = Array.from(document.querySelectorAll(INTERACTIVE));
            const typeable = (el) => {
                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute('role') || '';
                const type = (el.getAttribute('type') || '').toLowerCase();
                const noType = ['submit', 'button', 'reset', 'checkbox', 'radio', 'image', 'file'];
                return (tag === 'input' && !noType.includes(type)) || tag === 'textarea' || role === 'textbox' || role === 'combobox';
            };
            const visible = (el) => {
                const rect = el.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0 && window.getComputedStyle(el).visibility !== 'hidden';
            };
            const ordered = [...all.filter(el => visible(el) && typeable(el)), ...all.filter(el => visible(el) && !typeable(el))].slice(0, 50);
            const elements = [];
            ordered.forEach((el, idx) => {
                const eid = 'e' + idx;
                el.setAttribute('data-agent-id', eid);
                const tag = el.tagName.toLowerCase();
                let role = el.getAttribute('role') || tag;
                if (tag === 'a') role = 'link';
                let text = '';
                if (tag === 'input' || tag === 'textarea') {
                    text = el.getAttribute('value') || '';
                }
                if (!text) {
                    let raw = (el.textContent || '').trim().slice(0, 80);
                    if (!raw.includes('{') && !raw.includes(';')) text = raw;
                }
                const label = el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('name') || '';
                const inputType = el.getAttribute('type') || '';
                if (!text && !label && !inputType) return;
                elements.push({
                    element_id: eid,
                    role: role,
                    tag: tag,
                    text: text || label || inputType,
                    label: label,
                    enabled: !el.disabled,
                    input_type: inputType
                });
            });
            return elements;
        }""")

    @staticmethod
    def _format_elements(elements: List[Dict]) -> str:
        """Formats extracted elements into the prompt-friendly YAML-like string."""
        if not elements:
            return "(no interactive elements found)"
        lines = []
        for el in elements:
            tag = el.get('tag', '')
            role = el.get('role', '')
            input_type = el.get('input_type', '')

            # Build a clear type hint so the model knows what actions apply
            non_typeable_inputs = ('submit', 'button', 'reset', 'checkbox', 'radio', 'image', 'file')
            if (tag in ('input', 'textarea') or role in ('textbox', 'combobox')) and input_type not in non_typeable_inputs:
                kind = "textbox" if not input_type or input_type in ('text', 'search', '') else f"input[{input_type}]"
                action_hint = "USE TYPE to enter text"
            elif tag == 'select':
                kind = "dropdown"
                action_hint = "USE SELECT"
            elif input_type in non_typeable_inputs:
                kind = "button"
                action_hint = "USE CLICK"
            elif tag == 'a' or role == 'link':
                kind = "link"
                action_hint = "USE CLICK"
            else:
                kind = role or tag
                action_hint = "USE CLICK"

            parts = [
                f"  - element_id: {el['element_id']}",
                f"    kind: {kind}",
            ]
            if el.get('text'):
                parts.append(f'    text: "{el["text"]}"')
            if el.get('label'):
                parts.append(f'    label: "{el["label"]}"')
            parts.append(f"    enabled: {str(el['enabled']).lower()}")
            parts.append(f"    hint: {action_hint}")
            lines.append("\n".join(parts))
        return "\n".join(lines)
