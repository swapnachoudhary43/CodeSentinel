"""Build the CodeSentinel evaluation presentation."""

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

W, H = 13.333, 7.5
FONT = "Segoe UI"

INK = RGBColor(0x10, 0x12, 0x18)
TEXT = RGBColor(0x1C, 0x20, 0x2A)
MUTED = RGBColor(0x5E, 0x68, 0x7A)
FAINT = RGBColor(0x8B, 0x93, 0xA3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PAGE = RGBColor(0xF4, 0xF5, 0xF8)
LINE = RGBColor(0xE4, 0xE7, 0xEE)
DARK = RGBColor(0x0D, 0x0F, 0x14)
DARK_2 = RGBColor(0x17, 0x1A, 0x22)
ACCENT = RGBColor(0x6D, 0x5E, 0xFC)
ACCENT_SOFT = RGBColor(0xEF, 0xED, 0xFF)
ACCENT_INK = RGBColor(0x3C, 0x2F, 0xB8)
GREEN = RGBColor(0x12, 0x7A, 0x45)
GREEN_BG = RGBColor(0xE8, 0xF7, 0xEE)
AMBER = RGBColor(0x9A, 0x67, 0x00)
AMBER_BG = RGBColor(0xFF, 0xF6, 0xDF)
RED = RGBColor(0xC2, 0x41, 0x3B)
RED_BG = RGBColor(0xFF, 0xF1, 0xF0)

TOTAL = 23


def _anchor(tf, anchor):
    mapping = {
        MSO_ANCHOR.TOP: "t",
        MSO_ANCHOR.MIDDLE: "ctr",
        MSO_ANCHOR.BOTTOM: "b",
    }
    tf._txBody.bodyPr.set("anchor", mapping.get(anchor, "t"))


def set_run(run, size, color, bold=False, font=FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        node = rpr.find(qn(tag))
        if node is None:
            node = rpr.makeelement(qn(tag), {})
            rpr.append(node)
        node.set("typeface", font)


def add_text(slide, x, y, w, h, text, size=16, bold=False, color=TEXT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    _anchor(tf, anchor)
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_before = Pt(0)
    p.space_after = Pt(0)
    run = p.add_run()
    run.text = text
    set_run(run, size, color, bold, font)
    return shape


def add_paragraphs(slide, x, y, w, h, items, size=15, color=TEXT, bold=False, after=6):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    for index, item in enumerate(items):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_before = Pt(0)
        p.space_after = Pt(after)
        run = p.add_run()
        run.text = item
        set_run(run, size, color, bold)
    return shape


def rect(slide, x, y, w, h, fill, line=None, radius=0.08):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.adjustments[0] = radius
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    return shape


def bar(slide, x, y, w, h, fill):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    return shape


def bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def footer(slide, number, dark=False):
    color = RGBColor(0x8B, 0x93, 0xA3) if dark else FAINT
    add_text(slide, 0.55, 7.14, 8, 0.24, "CodeSentinel   ·   Evidence-driven multi-agent review", 11, color=color)
    add_text(slide, 10.6, 7.14, 2.2, 0.24, f"{number:02d}   /   {TOTAL:02d}", 11, color=color, align=PP_ALIGN.RIGHT)


def content_chrome(slide, kicker, title):
    bg(slide, PAGE)
    bar(slide, 0, 0, 0.08, 7.5, ACCENT)
    add_text(slide, 0.5, 0.28, 12, 0.24, kicker.upper(), 12, True, ACCENT)
    add_text(slide, 0.5, 0.52, 12.2, 0.48, title, 28, True, INK)
    footer(slide, content_chrome.n)
    content_chrome.n += 1


content_chrome.n = 2


def new(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def build():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    prs.core_properties.title = "CodeSentinel — Evidence-Driven Multi-Agent Intelligent Code Review System"
    prs.core_properties.subject = "M.Tech project evaluation presentation"
    prs.core_properties.category = "Advanced Multi-Agent Systems"

    title_slide(prs)
    agenda(prs)
    problem(prs)
    objectives(prs)
    principles(prs)
    context(prs)
    architecture(prs)
    workflow(prs)
    agents_one(prs)
    agents_two(prs)
    evidence(prs)
    languages(prs)
    human(prs)
    safety(prs)
    product(prs)
    stack(prs)
    example(prs)
    evaluation(prs)
    limits(prs)
    future(prs)
    sources(prs)
    conclusion(prs)
    close(prs)

    out = r"c:\Users\Swapna Choudhary\OneDrive\Desktop\CodeSentinel\CodeSentinel_Presentation.pptx"
    prs.save(out)
    return out


def title_slide(prs):
    s = new(prs)
    bg(s, DARK)
    bar(s, 0, 0, 0.12, 7.5, ACCENT)
    add_text(s, 0.7, 1.35, 10, 0.3, "M.TECH   ·   ADVANCED MULTI-AGENT SYSTEMS", 13, True, ACCENT)
    add_text(s, 0.68, 1.85, 11.5, 1.15, "CodeSentinel", 60, True, WHITE)
    add_text(
        s, 0.7, 3.15, 10.5, 1.1,
        "Evidence-Driven Multi-Agent\nIntelligent Code Review System",
        28, False, RGBColor(0xD5, 0xD8, 0xE2),
    )
    add_text(
        s, 0.7, 4.7, 9.5, 0.7,
        "A review platform that keeps tool output, model suggestions,\nand human judgment distinct.",
        16, False, FAINT,
    )
    rect(s, 0.7, 6.15, 2.4, 0.42, ACCENT, radius=0.5)
    add_text(s, 0.7, 6.22, 2.4, 0.3, "Project evaluation", 13, True, WHITE, align=PP_ALIGN.CENTER)
    add_text(s, 8.4, 6.22, 4.3, 0.3, "Local analysis   ·   No code execution", 13, False, FAINT, align=PP_ALIGN.RIGHT)


def agenda(prs):
    s = new(prs)
    content_chrome(s, "Overview", "How this presentation is organised")
    items = [
        ("01", "Problem", "Why mixed review signals create false confidence."),
        ("02", "Approach", "Evidence rules, shared workflow state, and a human decision."),
        ("03", "Architecture", "The product, the API, the agents, and the tools."),
        ("04", "Evaluation", "What was tested, what a sample review shows, and the limits."),
    ]
    for i, (num, title, body) in enumerate(items):
        y = 1.35 + i * 1.3
        rect(s, 0.55, y, 12.2, 1.12, WHITE, LINE, 0.1)
        add_text(s, 0.85, y + 0.28, 1.1, 0.5, num, 26, True, ACCENT)
        add_text(s, 2.2, y + 0.22, 9.8, 0.4, title, 20, True, INK)
        add_text(s, 2.2, y + 0.62, 9.8, 0.32, body, 15, False, MUTED)


def problem(prs):
    s = new(prs)
    content_chrome(s, "Problem", "Three signals are often treated as one fact")
    cards = [
        ("Tool output", "A linter or compiler reports a diagnostic. That is evidence about the program text, not a proof of runtime failure."),
        ("Model suggestion", "A language model can propose a defect. Without independent evidence, the proposal remains a possibility."),
        ("Human judgment", "A reviewer accepts, challenges, or defers. That records a decision. It does not certify the code."),
    ]
    for i, (title, body) in enumerate(cards):
        x = 0.55 + i * 4.15
        rect(s, x, 1.4, 3.95, 3.15, WHITE, LINE, 0.1)
        bar(s, x, 1.4, 3.95, 0.08, ACCENT)
        add_text(s, x + 0.28, 1.7, 3.4, 0.7, title, 20, True, INK)
        add_text(s, x + 0.28, 2.5, 3.4, 1.7, body, 15, False, MUTED)
    rect(s, 0.55, 4.75, 12.25, 1.95, INK, radius=0.1)
    add_text(s, 0.85, 5.0, 11.6, 0.4, "The failure mode", 16, True, ACCENT)
    add_text(
        s, 0.85, 5.45, 11.6, 0.95,
        "If a missing analyzer is reported as a pass, or an unverified suggestion is reported as a defect, the review looks complete when the evidence is not. CodeSentinel is built to prevent that collapse.",
        16, False, WHITE,
    )


def objectives(prs):
    s = new(prs)
    content_chrome(s, "Aim", "What the project sets out to do")
    add_text(
        s, 0.55, 1.25, 12.2, 0.7,
        "Coordinate specialized checks, record what each check could establish, and pause for a person when findings need a decision.",
        18, False, MUTED,
    )
    goals = [
        ("01", "Accept source", "Python, JavaScript, Java, C, and C++, within a bounded input size."),
        ("02", "Check without running", "Syntax and static checks use compilers and linters. Submitted code is not executed."),
        ("03", "Ask, then test", "Gemini proposes structured findings. A separate pass compares them with static evidence."),
        ("04", "Stay honest", "Unavailable tools stay unavailable. Incomplete analysis is named as incomplete."),
        ("05", "Leave a record", "Agent status, findings, and the human decision remain on the review thread."),
        ("06", "Support a demo", "A React workspace shows the real workflow, not a simulated progress bar."),
    ]
    for i, (num, title, body) in enumerate(goals):
        col, row = i % 3, i // 3
        x = 0.55 + col * 4.15
        y = 2.15 + row * 2.25
        rect(s, x, y, 3.95, 2.05, WHITE, LINE, 0.1)
        add_text(s, x + 0.24, y + 0.2, 1.2, 0.35, num, 14, True, ACCENT)
        add_text(s, x + 0.24, y + 0.58, 3.45, 0.4, title, 18, True, INK)
        add_text(s, x + 0.24, y + 1.08, 3.45, 0.75, body, 13, False, MUTED)


def principles(prs):
    s = new(prs)
    content_chrome(s, "Design", "Rules the workflow is not allowed to break")
    rules = [
        ("Evidence before confidence", "A finding carries a verification outcome. The interface does not upgrade a possibility into a confirmed defect."),
        ("Unavailable is not a pass", "If Ruff, Node, javac, or a C/C++ compiler cannot run, that check is UNAVAILABLE."),
        ("One reassessment", "Gemini may be asked once to reconsider unverified findings. The loop is bounded."),
        ("No invented defects", "The adversarial pass challenges existing claims. It does not add new bugs to make the review look thorough."),
        ("A decision is not a proof", "Approve, reject, defer, request changes, or approve with warnings. None of these certifies security or correctness."),
        ("Source is data", "Analysis commands are fixed tools. User source is passed as input, not as a shell command."),
    ]
    for i, (title, body) in enumerate(rules):
        y = 1.28 + i * 0.9
        rect(s, 0.55, y, 12.25, 0.8, WHITE, LINE, 0.12)
        bar(s, 0.55, y, 0.08, 0.8, ACCENT)
        add_text(s, 0.9, y + 0.1, 4.3, 0.55, title, 16, True, INK, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, 5.3, y + 0.1, 7.2, 0.6, body, 14, False, MUTED, anchor=MSO_ANCHOR.MIDDLE)


def context(prs):
    s = new(prs)
    content_chrome(s, "System", "From submitted source to a recorded decision")
    steps = [
        ("1", "Workspace", "The reviewer pastes source and selects a language."),
        ("2", "API", "FastAPI starts a review thread and streams real agent events."),
        ("3", "Workflow", "LangGraph runs the agents over one shared state."),
        ("4", "Evidence", "Tools, AST checks, and Gemini write findings with status."),
        ("5", "Decision", "A person records a decision. The report states the limits."),
    ]
    for i, (num, title, body) in enumerate(steps):
        x = 0.5 + i * 2.55
        rect(s, x, 1.7, 2.35, 2.7, WHITE, LINE, 0.12)
        oval = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.85), Inches(1.95), Inches(0.58), Inches(0.58))
        oval.fill.solid()
        oval.fill.fore_color.rgb = ACCENT
        oval.line.fill.background()
        add_text(s, x + 0.85, 2.06, 0.58, 0.38, num, 16, True, WHITE, align=PP_ALIGN.CENTER)
        add_text(s, x + 0.16, 2.7, 2.05, 0.4, title, 16, True, INK, align=PP_ALIGN.CENTER)
        add_text(s, x + 0.16, 3.15, 2.05, 1.0, body, 12, False, MUTED, align=PP_ALIGN.CENTER)
        if i < 4:
            add_text(s, x + 2.15, 2.05, 0.4, 0.4, "→", 18, True, ACCENT, align=PP_ALIGN.CENTER)
    rect(s, 0.55, 4.7, 12.25, 2.0, WHITE, LINE, 0.1)
    add_text(s, 0.85, 4.95, 11.6, 0.35, "Shared state, not a pile of scripts", 16, True, INK)
    add_text(
        s, 0.85, 5.4, 11.6, 1.0,
        "Every stage reads and updates the same review state: source, language, findings, errors, agent status, verification, and the human decision. Later stages can preserve earlier findings. They are not allowed to erase a failed check by reporting a clean pass.",
        15, False, MUTED,
    )


def architecture(prs):
    s = new(prs)
    content_chrome(s, "Architecture", "Four layers, each with a narrow job")
    layers = [
        ("Experience", "React and TypeScript", "Editor, language selection, live workflow events, finding filters, and the five review decisions."),
        ("Service", "FastAPI", "Analysis, streaming progress, review retrieval, and decision submission. Review threads are checkpointed in memory."),
        ("Orchestration", "LangGraph", "Ordered agents, conditional stops, one reassessment edge, and a pause when human review is required."),
        ("Analysis", "Tools and Gemini", "Ruff, language syntax tools, Python AST checks, and a schema-constrained Gemini review."),
    ]
    for i, (name, tech, body) in enumerate(layers):
        y = 1.32 + i * 1.35
        rect(s, 0.55, y, 12.25, 1.22, WHITE, LINE, 0.1)
        rect(s, 0.75, y + 0.28, 2.5, 0.66, ACCENT_SOFT, radius=0.15)
        add_text(s, 0.75, y + 0.28, 2.5, 0.4, name, 14, True, ACCENT_INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, 0.75, y + 0.62, 2.5, 0.28, tech, 11, False, ACCENT, align=PP_ALIGN.CENTER)
        add_text(s, 3.55, y + 0.28, 8.9, 0.72, body, 16, False, TEXT, anchor=MSO_ANCHOR.MIDDLE)


def workflow(prs):
    s = new(prs)
    content_chrome(s, "Workflow", "The path a review actually takes")
    left = [
        ("1", "Validate", "Non-empty source, size limit, supported language."),
        ("2", "Syntax", "Stop on a syntax defect or an unavailable checker."),
        ("3", "Context", "Record language. Parse Python structure when possible."),
        ("4", "Static analysis", "Ruff for Python. Other languages: linter unavailable."),
        ("5", "Specialized checks", "Logic, security, and performance, each with its own scope."),
        ("6", "Edge cases", "Test ideas derived only from findings already produced."),
    ]
    right = [
        ("7", "AI review", "Gemini returns structured findings and a summary."),
        ("8", "Verification", "Compare claims with static evidence."),
        ("9", "Reassessment", "At most one retry when a model finding is unresolved."),
        ("10", "Adversarial pass", "Challenge weak claims. Add no new defects."),
        ("11", "Human review", "Pause when findings exist and review is requested."),
        ("12", "Report", "Summarize results and the limits of the run."),
    ]
    for column, items, x in ((left, left, 0.5), (right, right, 6.9)):
        for i, (num, title, body) in enumerate(items):
            y = 1.28 + i * 0.9
            rect(s, x, y, 5.95, 0.8, WHITE, LINE, 0.12)
            add_text(s, x + 0.18, y + 0.18, 0.55, 0.45, num, 16, True, ACCENT, anchor=MSO_ANCHOR.MIDDLE)
            add_text(s, x + 0.8, y + 0.08, 4.9, 0.32, title, 15, True, INK)
            add_text(s, x + 0.8, y + 0.4, 4.9, 0.32, body, 12, False, MUTED)


def agents_one(prs):
    s = new(prs)
    content_chrome(s, "Agents", "What the first group is responsible for")
    cards = [
        ("Input and context", "Checks the submission and, for Python, lists imports, functions, and classes. Other languages are marked as having no structural parser."),
        ("Syntax checker", "Uses Python’s compiler, Node’s syntax check, javac, or a C/C++ compiler in syntax-only mode. It does not run the program."),
        ("Static analysis", "Runs Ruff on Python and keeps each diagnostic. If no linter exists for the language, the agent reports UNAVAILABLE."),
        ("Logic", "Inspects Python syntax trees for division. A division is a risk to review, not proof that the divisor is zero."),
        ("Security", "Looks for a small set of patterns: dynamic SQL, shell execution, hard-coded secrets, and unsafe deserialization."),
        ("Performance", "Flags nested Python loops as a possible cost. It does not measure runtime."),
    ]
    for i, (title, body) in enumerate(cards):
        col, row = i % 3, i // 3
        x = 0.5 + col * 4.2
        y = 1.35 + row * 2.7
        rect(s, x, y, 4.0, 2.5, WHITE, LINE, 0.1)
        add_text(s, x + 0.24, y + 0.22, 3.5, 0.55, title, 18, True, INK)
        add_text(s, x + 0.24, y + 0.9, 3.5, 1.35, body, 14, False, MUTED)


def agents_two(prs):
    s = new(prs)
    content_chrome(s, "Agents", "Review, challenge, decision, and report")
    cards = [
        ("Edge cases", "Suggests scenarios only when a finding gives a reason, such as a division or an injection pattern. A file with no findings does not receive a generic test list."),
        ("AI review", "Gemini reviews the source against a fixed response schema. Findings are potential until verification. The prompt tells the model not to invent execution results."),
        ("Verification", "Assigns VERIFIED, CONTRADICTED, UNRESOLVED, or NOT_CHECKED. A literal division by zero can be verified. An ordinary division stays unresolved."),
        ("Adversarial review", "Reads the findings after verification and records challenges. It is not a second model invited to discover extra bugs."),
        ("Human review", "Interrupts the graph when findings exist. The API stores one of five decisions and can return that state later in the same process."),
        ("Report", "Writes the summary the reviewer sees, including tool limits, the Gemini summary, and the recorded decision."),
    ]
    for i, (title, body) in enumerate(cards):
        col, row = i % 3, i // 3
        x = 0.5 + col * 4.2
        y = 1.35 + row * 2.7
        rect(s, x, y, 4.0, 2.5, WHITE, LINE, 0.1)
        add_text(s, x + 0.24, y + 0.22, 3.5, 0.5, title, 18, True, INK)
        add_text(s, x + 0.24, y + 0.85, 3.5, 1.4, body, 14, False, MUTED)


def evidence(prs):
    s = new(prs)
    content_chrome(s, "Evidence", "Status words the system uses consistently")
    add_text(s, 0.55, 1.22, 6, 0.3, "Tool status", 14, True, MUTED)
    statuses = [
        (GREEN_BG, GREEN, "PASSED", "The check ran and reported no issue."),
        (AMBER_BG, AMBER, "FAILED", "The check ran and reported a problem."),
        (RED_BG, RED, "UNAVAILABLE", "The check could not run. This is not a pass."),
    ]
    for i, (fill, color, name, body) in enumerate(statuses):
        x = 0.55 + i * 4.15
        rect(s, x, 1.58, 3.95, 1.45, fill, radius=0.1)
        add_text(s, x + 0.22, 1.74, 3.5, 0.35, name, 16, True, color)
        add_text(s, x + 0.22, 2.15, 3.5, 0.65, body, 14, False, TEXT)
    add_text(s, 0.55, 3.25, 8, 0.3, "Verification outcome", 14, True, MUTED)
    outcomes = [
        ("VERIFIED", "Static evidence confirms the specific claim, such as a compiler diagnostic or a literal division by zero."),
        ("CONTRADICTED", "The claim does not match the parsed source. A division warning with no division is contradicted."),
        ("UNRESOLVED", "The pattern exists, but the dangerous case is not proven. A variable divisor is the usual example."),
        ("NOT_CHECKED", "No independent rule confirmed or refuted the finding. A Ruff diagnostic remains a tool report."),
    ]
    for i, (name, body) in enumerate(outcomes):
        col, row = i % 2, i // 2
        x = 0.55 + col * 6.35
        y = 3.65 + row * 1.45
        rect(s, x, y, 6.15, 1.32, WHITE, LINE, 0.1)
        add_text(s, x + 0.22, y + 0.16, 5.7, 0.32, name, 15, True, ACCENT_INK)
        add_text(s, x + 0.22, y + 0.52, 5.7, 0.65, body, 13, False, MUTED)


def languages(prs):
    s = new(prs)
    content_chrome(s, "Coverage", "Each language is analyzed only as far as its tools go")
    headers = ["Language", "Syntax", "Static analysis", "Deeper structure"]
    rows = [
        ["Python", "Built-in compiler", "Ruff", "AST: structure, division, nested loops"],
        ["JavaScript", "Node syntax check", "No linter configured", "Not parsed as Python"],
        ["Java", "javac, if installed", "No linter configured", "Not parsed as Python"],
        ["C", "gcc or clang, syntax only", "No linter configured", "Not parsed as Python"],
        ["C++", "g++ or clang++, syntax only", "No linter configured", "Not parsed as Python"],
    ]
    xs = [0.55, 2.7, 6.15, 9.15]
    widths = [2.05, 3.35, 2.9, 3.6]
    rect(s, 0.55, 1.35, 12.25, 0.62, INK, radius=0.08)
    for x, w, label in zip(xs, widths, headers):
        add_text(s, x + 0.12, 1.48, w - 0.1, 0.36, label, 13, True, WHITE)
    for r, row in enumerate(rows):
        y = 2.08 + r * 0.85
        fill = WHITE if r % 2 == 0 else RGBColor(0xF8, 0xF9, 0xFC)
        rect(s, 0.55, y, 12.25, 0.78, fill, LINE, 0.06)
        for x, w, value in zip(xs, widths, row):
            add_text(s, x + 0.12, y + 0.18, w - 0.15, 0.45, value, 13, x == xs[0], INK if x == xs[0] else TEXT, anchor=MSO_ANCHOR.MIDDLE)


def human(prs):
    s = new(prs)
    content_chrome(s, "Human review", "The workflow pauses. It does not decide for the reviewer")
    decisions = [
        ("Approve", "Accept the review as presented."),
        ("Approve with warnings", "Accept it while keeping the concerns visible."),
        ("Request changes", "Ask for the code or the findings to be revisited."),
        ("Reject", "Decline the submission on the recorded evidence."),
        ("Defer", "Leave the decision open without pretending it is resolved."),
    ]
    for i, (title, body) in enumerate(decisions):
        x = 0.5 + i * 2.55
        rect(s, x, 1.4, 2.4, 2.35, WHITE, LINE, 0.12)
        add_text(s, x + 0.16, 1.6, 2.08, 0.7, title, 16, True, INK)
        add_text(s, x + 0.16, 2.4, 2.08, 1.05, body, 13, False, MUTED)
    rect(s, 0.55, 4.0, 12.25, 2.7, WHITE, LINE, 0.1)
    points = [
        "Review starts only when findings exist and human review is requested.",
        "The decision, optional feedback, and thread identifier are stored in the running API.",
        "Refreshing the review returns the same decision. It is not only a message on the screen.",
        "The report states that the decision records judgment. It does not prove the code is correct or secure.",
    ]
    add_text(s, 0.85, 4.2, 11.6, 0.4, "What “stored” means in this build", 16, True, INK)
    for i, point in enumerate(points):
        add_text(s, 0.85, 4.75 + i * 0.42, 11.6, 0.4, "•   " + point, 15, False, TEXT)


def safety(prs):
    s = new(prs)
    content_chrome(s, "Trust boundary", "The backend analyzes source. It does not run it")
    items = [
        ("No program execution", "Python is compiled for syntax. C and C++ are checked with -fsyntax-only. Java is compiled only as a check, in a temporary directory, and the program is not launched."),
        ("No shell from user text", "Tool commands are fixed argument lists. Source reaches them through standard input or a temporary file, not through a shell string."),
        ("Bounded model output", "Gemini must return JSON matching the review schema. Empty or invalid responses fail the AI step and preserve earlier findings."),
        ("Visible incompleteness", "A missing key, a failed model call, or a missing tool produces a partial result and the phrase INSUFFICIENT EVIDENCE — ANALYSIS INCOMPLETE."),
        ("Secret handling", "The API key stays in the server environment. If an error message contains the key, the client response redacts it."),
        ("Audit trail", "Each agent appends a timestamped status. The interface shows that log instead of inventing stages while it waits."),
    ]
    for i, (title, body) in enumerate(items):
        col, row = i % 2, i // 2
        x = 0.5 + col * 6.4
        y = 1.3 + row * 1.8
        rect(s, x, y, 6.15, 1.65, WHITE, LINE, 0.1)
        add_text(s, x + 0.24, y + 0.18, 5.7, 0.35, title, 16, True, INK)
        add_text(s, x + 0.24, y + 0.6, 5.7, 0.85, body, 13, False, MUTED)


def product(prs):
    s = new(prs)
    content_chrome(s, "Product", "A workspace a reviewer can actually use")
    features = [
        ("Language workspace", "Switching language loads an example in that language and clears the previous result."),
        ("Editor", "Line count, character count, and a clear statement that code is analyzed, not executed."),
        ("Live progress", "The page reads workflow events from the API stream. It does not advance stages on a timer."),
        ("Findings", "Filters separate tool reports from potential issues. Each card shows evidence and verification."),
        ("Review controls", "All five decisions are available while a review is pending, with optional written feedback."),
        ("Recovery", "A failed connection is shown as a failure. The screen does not fill in a successful report."),
    ]
    for i, (title, body) in enumerate(features):
        col, row = i % 3, i // 3
        x = 0.5 + col * 4.2
        y = 1.4 + row * 2.55
        rect(s, x, y, 4.0, 2.35, WHITE, LINE, 0.1)
        add_text(s, x + 0.24, y + 0.28, 3.5, 0.55, title, 18, True, INK)
        add_text(s, x + 0.24, y + 0.95, 3.5, 1.1, body, 14, False, MUTED)


def stack(prs):
    s = new(prs)
    content_chrome(s, "Technology", "The stack is the one the repository runs")
    groups = [
        ("Workflow", ["Python 3.13", "LangGraph", "Pydantic", "In-memory checkpoints"]),
        ("Service", ["FastAPI", "Uvicorn", "Server-sent events", "CORS for the local app"]),
        ("Analysis", ["Ruff", "Python AST", "Node, javac, gcc/clang", "Google Gemini"]),
        ("Product", ["React 19", "TypeScript", "Vite", "Satoshi and JetBrains Mono"]),
        ("Tests", ["pytest", "FastAPI TestClient", "Vitest", "Playwright"]),
    ]
    for i, (title, items) in enumerate(groups):
        x = 0.45 + i * 2.56
        rect(s, x, 1.45, 2.42, 4.9, WHITE, LINE, 0.1)
        rect(s, x + 0.18, 1.68, 2.06, 0.48, ACCENT_SOFT, radius=0.2)
        add_text(s, x + 0.18, 1.76, 2.06, 0.34, title, 13, True, ACCENT_INK, align=PP_ALIGN.CENTER)
        for j, item in enumerate(items):
            add_text(s, x + 0.2, 2.5 + j * 0.75, 2.05, 0.6, item, 14, False, INK)


def example(prs):
    s = new(prs)
    content_chrome(s, "Walkthrough", "A short Python function, read honestly")
    rect(s, 0.5, 1.35, 5.5, 5.3, INK, radius=0.1)
    add_text(s, 0.75, 1.55, 5, 0.3, "main.py", 12, True, FAINT)
    code = "def divide(a, b):\n    return a / b\n\ndef calculate(values):\n    total = 0\n    for value in values:\n        total += value\n    return total"
    add_text(s, 0.75, 2.0, 5, 4.2, code, 16, False, WHITE, font="Consolas", anchor=MSO_ANCHOR.MIDDLE)
    notes = [
        ("Logic agent", "Reports the division. The divisor is a parameter, so zero is possible, not proven."),
        ("Gemini", "May find no separate actionable issue. That does not delete the logic finding."),
        ("Verification", "The division stays UNRESOLVED. A claim of division in code that has none would be CONTRADICTED."),
        ("Reviewer", "Can approve with warnings, request changes, reject, or defer. The choice is stored with the thread."),
    ]
    for i, (title, body) in enumerate(notes):
        y = 1.35 + i * 1.32
        rect(s, 6.25, y, 6.55, 1.2, WHITE, LINE, 0.1)
        add_text(s, 6.5, y + 0.12, 6.1, 0.3, title, 14, True, ACCENT_INK)
        add_text(s, 6.5, y + 0.46, 6.1, 0.6, body, 13, False, TEXT)


def evaluation(prs):
    s = new(prs)
    content_chrome(s, "Evaluation", "The claims below were exercised, not assumed")
    rows = [
        ("API", "Health, valid analysis, missing fields, empty and oversized input, an unsupported language, a missing review, and an invalid decision."),
        ("Workflow", "Syntax failures stop later agents. Ruff findings survive a Gemini failure. Reassessment runs at most once. Logic findings are retained."),
        ("Evidence", "Division without a literal zero is unresolved. A tool report is not treated as an independently verified runtime defect."),
        ("Failure", "A missing Gemini key and a missing Ruff executable produce a partial result, not a pass."),
        ("Interface", "Empty input does not call the API. One analysis request is sent. Filters, review decisions, and network failure are covered."),
        ("Browser", "A full local run opens the workspace, completes analysis, and stores a human decision."),
    ]
    for i, (name, body) in enumerate(rows):
        y = 1.28 + i * 0.9
        rect(s, 0.5, y, 12.3, 0.8, WHITE, LINE, 0.1)
        rect(s, 0.68, y + 0.18, 1.7, 0.44, ACCENT_SOFT, radius=0.2)
        add_text(s, 0.68, y + 0.24, 1.7, 0.32, name, 12, True, ACCENT_INK, align=PP_ALIGN.CENTER)
        add_text(s, 2.6, y + 0.16, 9.9, 0.52, body, 13, False, TEXT, anchor=MSO_ANCHOR.MIDDLE)


def limits(prs):
    s = new(prs)
    content_chrome(s, "Limits", "What this build does not establish")
    items = [
        ("Python-centred depth", "Structural parsing, division checks, and nested-loop checks use the Python AST. Other languages do not receive those agents."),
        ("Narrow security checks", "The security agent matches patterns. It is not a taint analyzer and it does not implement the OWASP catalog."),
        ("Verification is partial", "Most runtime questions — nullness, bounds, intent — cannot be settled from syntax alone."),
        ("Model dependence", "AI review requires a configured Gemini key and a successful response. The model can still be wrong."),
        ("Process memory", "Review threads live in the API process. Restarting the server clears them."),
        ("No execution oracle", "Because user code is not run, the system cannot confirm a bug by reproducing it."),
    ]
    for i, (title, body) in enumerate(items):
        col, row = i % 3, i // 3
        x = 0.5 + col * 4.2
        y = 1.4 + row * 2.6
        rect(s, x, y, 4.0, 2.4, WHITE, LINE, 0.1)
        add_text(s, x + 0.24, y + 0.25, 3.5, 0.6, title, 17, True, INK)
        add_text(s, x + 0.24, y + 1.0, 3.5, 1.1, body, 14, False, MUTED)


def future(prs):
    s = new(prs)
    content_chrome(s, "Next", "Work that would deepen the same design")
    items = [
        ("01", "Data-flow evidence", "Track whether a divisor, index, or reference can actually reach the dangerous value."),
        ("02", "Language linters", "Add a real static analyzer for JavaScript, Java, C, and C++, and keep UNAVAILABLE when it is absent."),
        ("03", "Durable reviews", "Store threads outside process memory so a decision survives a restart."),
        ("04", "Stronger challenge", "Let the adversarial pass request specific evidence, still without inventing findings."),
        ("05", "Review in the toolchain", "Connect the same API to an editor or a pull request, without weakening the evidence rules."),
    ]
    for i, (num, title, body) in enumerate(items):
        y = 1.32 + i * 1.05
        add_text(s, 0.55, y, 0.8, 0.7, num, 20, True, ACCENT, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, 1.5, y, 4.2, 0.7, title, 18, True, INK, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, 5.8, y, 6.9, 0.8, body, 15, False, MUTED, anchor=MSO_ANCHOR.MIDDLE)


def sources(prs):
    s = new(prs)
    content_chrome(s, "Sources", "Work and tools this design draws on")
    refs = [
        ("Static analysis practice", "Johnson, S. C. Lint, a C Program Checker. Bell Laboratories, 1978.\nSadowski, C., Aftandilian, E., Eagle, A., Miller-Cushon, L., and Jaspan, C. Lessons from Building Static Analysis Tools at Google. Communications of the ACM, 2018."),
        ("Security vocabulary", "OWASP Foundation. OWASP Top 10. https://owasp.org/www-project-top-ten/\nThe security agent uses a few related patterns. It does not implement this catalog."),
        ("Tools in the repository", "Astral. Ruff. https://docs.astral.sh/ruff/\nLangChain. LangGraph. https://docs.langchain.com/oss/python/langgraph/\nGoogle. Gemini API. https://ai.google.dev/gemini-api/docs\nRamírez, S. FastAPI. https://fastapi.tiangolo.com/"),
    ]
    for i, (title, body) in enumerate(refs):
        y = 1.4 + i * 1.75
        rect(s, 0.55, y, 12.25, 1.6, WHITE, LINE, 0.1)
        add_text(s, 0.85, y + 0.16, 11.6, 0.32, title, 16, True, INK)
        add_paragraphs(s, 0.85, y + 0.54, 11.6, 0.9, body.split("\n"), size=14, color=MUTED, after=2)


def conclusion(prs):
    s = new(prs)
    bg(s, DARK)
    bar(s, 0, 0, 0.12, 7.5, ACCENT)
    add_text(s, 0.7, 0.45, 8, 0.3, "CONCLUSION", 13, True, ACCENT)
    add_text(s, 0.7, 1.05, 11.5, 1.6, "A review is only as\nstrong as its evidence.", 40, True, WHITE)
    points = [
        "Agents gather different kinds of evidence and are required to expose their limits.",
        "Model output is kept separate from tool output and from the reviewer’s decision.",
        "The product shows that workflow, so an evaluation can inspect the result rather than a mock.",
    ]
    for i, point in enumerate(points):
        y = 3.5 + i * 0.85
        rect(s, 0.7, y, 11.8, 0.72, DARK_2, radius=0.1)
        add_text(s, 0.95, y + 0.16, 11.3, 0.42, point, 16, False, WHITE, anchor=MSO_ANCHOR.MIDDLE)
    footer(s, 22, dark=True)


def close(prs):
    s = new(prs)
    bg(s, DARK)
    bar(s, 0, 0, 0.12, 7.5, ACCENT)
    add_text(s, 0.7, 1.7, 11, 0.4, "DISCUSSION", 14, True, ACCENT)
    add_text(s, 0.7, 2.2, 11, 1.2, "Thank you.", 60, True, WHITE)
    add_text(
        s, 0.7, 3.7, 10, 1.0,
        "Questions are welcome on the agent boundaries,\nthe evidence rules, and what the system refuses to claim.",
        20, False, RGBColor(0xD5, 0xD8, 0xE2),
    )
    add_text(s, 0.7, 5.5, 10, 0.8, "CodeSentinel\nEvidence-Driven Multi-Agent Intelligent Code Review System", 16, False, FAINT)
    footer(s, 23, dark=True)


if __name__ == "__main__":
    print(build())
