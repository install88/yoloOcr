"""
Shared date regex patterns — used by baseline_rec.py, convert_to_yolo.py, pipeline.py.
Derived from full analysis of train_label.txt + val_label.txt.

Format distribution in dataset:
  YYYY.MM.DD          887  (2026.05.01, BBF:2026.07.23 BD01)
  YYYYMMDD 8碼        505  (有效20270301, EXP20281103)
  DD.MM.YYYY          239  (30.05.2026, DATE:27.11.2025)
  YYYY MM DD 空格     121  (T2026 04 27 AK08, 2026 04 08)
  含數字其他           107  (西元2028年11月9日, 2028年3月9日)
  YY.MM.DD/民國年      84  (有效日期26.08.11, 製造115.03.30)
  只有年月             76  (09-2026, 2026.08, 賞味期限2026.09)
  YYMMDD 6碼           48  (261113, 260309AB)
  DD MM YYYY 空格      21  (23 02 2027, BB: 17 11 2026)
  YYYY/MM/DD            7  (2025-12-23)
  無數字              218  (有效日期, 賞味期限 — 純標籤行，不含日期)
"""
import re

# ── 完整日期（年月日）──────────────────────────────
_FULL_DATE = (
    # YYYY.MM.DD / YYYY/MM/DD / YYYY-MM-DD (逗號也算，OCR常把句點識為逗號)
    r"\d{4}[./\-,]\d{1,2}[./\-,]\d{1,2}"
    # DD.MM.YYYY / DD/MM/YYYY
    r"|\d{1,2}[./\-,]\d{1,2}[./\-,]\d{4}"
    # YYYY MM DD (空格分隔，前後可有非數字)
    r"|\d{4}\s+\d{1,2}\s+\d{1,2}"
    # DD MM YYYY (空格分隔)
    r"|\d{1,2}\s+\d{1,2}\s+\d{4}"
    # YYYYMMDD (8碼，19xx/20xx開頭，允許後面有1個多餘數字——OCR誤插)
    r"|(?<!\d)(?:20|19)\d{6}(?!\d{2})"
    # YYYYMMDD (8碼連續，後可接非數字)
    r"|\d{8}(?!\d)"
    # YYYYMM.DD (6碼+分隔+1-2碼，如 202602.10)
    r"|\d{6}[./\-]\d{1,2}(?!\d)"
    # YYMMDD (6碼連續，如 250925、261113)
    r"|\d{6}(?!\d)"
    # YY.MM.DD 或 民國年 1XX.MM.DD
    r"|\d{2,3}[./\-]\d{1,2}[./\-]\d{2,4}"
    # 中文完整: 2028年3月9日 / 西元2028年11月9日
    r"|\d{2,4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
)

# ── 只有年月（無日）──────────────────────────────
_YEAR_MONTH = (
    # YYYY.MM / YYYY-MM / YYYY/MM (只有年月)
    r"\d{4}[./\-]\d{1,2}(?![./\-\d])"
    # MM-YYYY / MM/YYYY
    r"|\d{1,2}[./\-]\d{4}(?!\d)"
    # YYMMDD 6碼 (兼容)
    r"|\d{6}(?!\d)"
)

# ── 帶前綴的格式 ──────────────────────────────────
_PREFIXED = (
    r"(?:有效|製造|到期|賞味|生産|出廠)"
    r"(?:日期|期限|期|日)?"
    r"[：:\s]*"
    r"\d{2}"
)

# 合併為單一 pattern（用於 is_date_text 判斷）
DATE_PATTERN = re.compile(
    f"(?:{_FULL_DATE}|{_YEAR_MONTH}|{_PREFIXED})",
    re.IGNORECASE,
)

# 用於最終抽取日期字串（只抓完整日期，不抓純年月）
EXTRACT_PATTERN = re.compile(
    f"(?:{_FULL_DATE})",
    re.IGNORECASE,
)


def is_date_text(text: str) -> bool:
    """判斷 transcription 是否為日期相關文字（用於過濾訓練資料）。"""
    if not text or text == "###":
        return False
    return bool(DATE_PATTERN.search(text))


def extract_date(text: str) -> str | None:
    """從辨識結果中抽出日期字串。回傳第一個匹配，或 None。"""
    m = EXTRACT_PATTERN.search(text)
    return m.group().strip() if m else None


# ── 快速自測 ──────────────────────────────────────
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    test_cases = [
        # 應該要 match
        ("製造日期:2025.08.14",     True,  "2025.08.14"),
        ("有效20270301",            True,  "20270301"),
        ("T2026 04 27 AK08",       True,  "2026 04 27"),
        ("有效日期26.08.11",        True,  "26.08.11"),
        ("製造115.03.30.",          True,  "115.03.30"),
        ("23 02 2027",             True,  "23 02 2027"),
        ("BB: 17 11 2026",         True,  "17 11 2026"),
        ("西元2028年11月9日",       True,  "2028年11月9日"),
        ("2028年3月9日",            True,  "2028年3月9日"),
        ("09-2026",                True,  None),   # 只有年月，不抽取
        ("2026.08",                True,  None),   # 只有年月
        ("EXP:10/2026",            True,  None),   # 只有年月
        ("DATE:27.11.2025",        True,  "27.11.2025"),
        ("BBF:2026.07.23 BD01",    True,  "2026.07.23"),
        ("2026.05.03 DH00577",     True,  "2026.05.03"),
        ("製造202602.10",           True,  "202602.10"),  # YYYYMM.DD
        ("250925 010 11:22",       True,  "250925"),      # YYMMDD+lot
        # 不應該 match
        ("有效日期",               False, None),
        ("賞味期限",               False, None),
        ("BESTBEFORE",            False, None),
        ("有效日期年月日",          False, None),
    ]

    passed = failed = 0
    for text, expect_match, expect_extract in test_cases:
        got_match = is_date_text(text)
        got_extract = extract_date(text)
        ok_m = got_match == expect_match
        ok_e = (expect_extract is None) or (got_extract and expect_extract in got_extract)
        if ok_m and ok_e:
            passed += 1
            print(f"  ✓  {text}")
        else:
            failed += 1
            print(f"  ✗  {text}")
            if not ok_m:
                print(f"       match: expect={expect_match} got={got_match}")
            if not ok_e:
                print(f"       extract: expect={expect_extract!r} got={got_extract!r}")

    print(f"\n{passed} passed, {failed} failed")
