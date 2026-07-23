import gettext
import os
import re

from utils.logging_setup import get_logger
from utils.utils import Utils

logger = get_logger("translations")


_locale = os.environ['LANG'] if "LANG" in os.environ else None
if not _locale or _locale == '':
    _locale = Utils.get_default_user_language()
elif _locale is not None and "_" in _locale:
    _locale = _locale[:_locale.index("_")]

class I18N:
    localedir = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'locale')
    locale = "en"
    translate = gettext.translation('base', localedir, languages=[_locale])

    @staticmethod
    def install_locale(locale, verbose=True):
        I18N.locale = locale
        I18N.translate = gettext.translation('base', I18N.localedir, languages=[locale], fallback=True)
        I18N.translate.install()
        if verbose:
            logger.info("Switched locale to: " + locale)

    @staticmethod
    def _(s):
        # return gettext.gettext(s)
        try:
            return I18N.translate.gettext(s)
        except KeyError:
            return s


def compare_running_warn(action: str) -> str:
    return I18N._("Compare is running, action not available: {0}").format(action)


def marks_transfer_running_warn(action: str) -> str:
    return I18N._("Marked files are being moved or copied, action not available: {0}").format(action)

    '''
    NOTE when gathering the translation strings, set _() == to gettext.gettext() instead of the above, and run:

        ```python C:\\Python310\\Tools\\i18n\\pygettext.py -d base -o locale\\base.pot .```

    in the base directory. The POT output file can be used as source for the PO files in each locale.
    Run personal script C:\\Scripts\\i18n_manager.py to generate new PO files and look for invalid translations.

    Bonus command:
        ```git diff Weidr\\locale\\de\\LC_MESSAGES\\base.po Weidr\\locale\\de\\LC_MESSAGES\\base1.po | rg -v "^.*#" | rg -C 3 "^(-|\\+)"```

    Then for each locale once the PO files are set up as desired, run below in the deepest locale directory to produce the MO file from the PO file:
        ```python C:\\Python310\\Tools\\i18n\\msgfmt.py -o base.mo base```
    '''

_ = I18N._

# Modifier names only — avoid short verbs like "Delete" (already msgid → "Löschen").
# Keep QKeySequence("Ctrl+…") bindings in English; use this for UI labels only.
_SHORTCUT_MODIFIER_RE = re.compile(r"\b(?:Ctrl|Shift|Alt)\b")


def format_shortcut(text: str) -> str:
    """Localize modifier names in a shortcut chord for display (e.g. Ctrl→Strg)."""
    # Literal _() calls so pygettext extracts these as keyboard keycap labels.
    modifiers = {
        "Ctrl": _("Ctrl"),
        "Shift": _("Shift"),
        "Alt": _("Alt"),
    }
    return _SHORTCUT_MODIFIER_RE.sub(lambda m: modifiers[m.group(0)], text)