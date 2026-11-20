"""Read the text on an extra plate. Readers only *transcribe*; plates.parse() turns the text into rules, so every
reader is judged by the same tested parser.

    tesseract  offline OCR. Works on the car's Wi-Fi (which has no internet). Needs the tesseract program:
               apt install tesseract-ocr tesseract-ocr-deu   (dev container) / brew install tesseract (Mac),
               and pip install pytesseract.
    claude     Claude vision (claude-opus-5-5). Much better on small, blurry or tilted plates, but needs internet
               (a second network on the laptop, e.g. a phone hotspot or a LAN cable) and an API key:
               pip install anthropic; export ANTHROPIC_API_KEY=... (or `ant auth login`).
               Each read is one API call; the mapper reads a plate only a few times, then stops.
    fixed      always returns the given text (simulation, tests).
"""
import base64

import cv2

PROMPT = (
    'This image shows a German traffic sign with one or more additional panels (Zusatzzeichen) below it. '
    'Transcribe the text on the additional panels exactly as written, one line per line of text, '
    'keeping numbers, dashes, dots and abbreviations (for example "Mo-Fr", "8-18 h", "max. 2 Std."). '
    'Do not interpret or translate. If the panels have no readable text, answer exactly: NONE'
)


class TesseractReader:
    name = 'tesseract'

    def __init__(self, log=print):
        import pytesseract
        pytesseract.get_tesseract_version()   # raises if the tesseract program is not installed
        self.tess = pytesseract
        langs = set(pytesseract.get_languages(config=''))
        self.lang = 'deu+eng' if 'deu' in langs else 'eng'
        self.log = log

    def __call__(self, sign_and_plate, plate):
        """plate: BGR/gray crop of the plate(s) only. Returns the text or None."""
        g = plate if plate.ndim == 2 else cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        scale = max(1.0, 600 / g.shape[1])   # OCR wants big letters: upscale small plates
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, g = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        g = cv2.copyMakeBorder(g, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        text = self.tess.image_to_string(g, lang=self.lang, config='--psm 6').strip()
        return text or None


class ClaudeReader:
    name = 'claude'
    MODEL = 'claude-opus-5-5'

    def __init__(self, log=print):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()   # API key / `ant auth login` profile from the environment
        self.log = log
        self.disabled = False

    def __call__(self, sign_and_plate, plate):
        """sign_and_plate: crop of the sign with its plates (the sign gives context). Returns the text or None."""
        if self.disabled:
            return None
        ok, jpg = cv2.imencode('.jpg', sign_and_plate, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        a = self.anthropic
        try:
            response = self.client.beta.messages.create(
                model=self.MODEL,
                max_tokens=1024,
                output_config={'effort': 'low'},          # transcription: no deep reasoning needed
                betas=['server-side-fallback-2026-07-01'],
                fallbacks='default',                       # if the model declines, the API retries on a fallback
                messages=[{'role': 'user', 'content': [
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                                 'data': base64.standard_b64encode(jpg.tobytes()).decode()}},
                    {'type': 'text', 'text': PROMPT},
                ]}],
            )
        except (a.AuthenticationError, a.PermissionDeniedError) as e:
            self.disabled = True                           # will not fix itself: stop trying
            self.log(f'claude reader disabled: {e.message}')
            return None
        except a.RateLimitError:
            self.log('claude reader: rate limited, will retry on a later sighting')
            return None
        except a.APIStatusError as e:
            self.log(f'claude reader: API error {e.status_code}: {e.message}')
            return None
        except a.APIConnectionError:
            self.log('claude reader: no connection to the API (the car Wi-Fi has no internet)')
            return None
        if response.stop_reason == 'refusal':
            return None
        text = '\n'.join(b.text for b in response.content if b.type == 'text').strip()
        return None if not text or text.upper() == 'NONE' else text


class FixedReader:
    name = 'fixed'

    def __init__(self, text, log=print):
        self.text = text

    def __call__(self, sign_and_plate, plate):
        return self.text or None


def make_reader(kind, log=print, fixed_text=''):
    """'tesseract' | 'claude' | 'fixed' | 'none' -> reader or None. Raises with a helpful message if unavailable."""
    if kind in ('', 'none'):
        return None
    if kind == 'fixed':
        return FixedReader(fixed_text, log)
    try:
        return {'tesseract': TesseractReader, 'claude': ClaudeReader}[kind](log)
    except KeyError:
        raise ValueError(f'unknown plate reader {kind!r} (tesseract, claude, fixed, none)') from None
    except ImportError as e:
        pkg = {'tesseract': 'pytesseract', 'claude': 'anthropic'}[kind]
        raise RuntimeError(f'plate reader {kind!r} needs `pip install {pkg}` ({e})') from None
    except Exception as e:   # pytesseract raises its own error when the tesseract program is missing
        raise RuntimeError(f'plate reader {kind!r} is not usable: {e}. See parking_rules/readers.py') from None
