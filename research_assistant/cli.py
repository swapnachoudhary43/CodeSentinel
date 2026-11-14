import argparse
from pathlib import Path

from config import get_gemini_api_key

from research_assistant.corpus import load_papers, load_sample_papers
from research_assistant.llm import ExtractiveModel, GeminiModel
from research_assistant.pipeline import answer, answer_ablation, compare, literature_row
from research_assistant.retrieve import HybridIndex
from research_assistant.types import ABSTAIN_MESSAGE, ABLATIONS, SYSTEMS, AnswerResult


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the generative research assistant on two local fixture notes. "
            "This corpus is not Qasper and extractive mode is not a Gemini result."
        )
    )
    parser.add_argument("--question", required=True, help="Question to ask the indexed notes.")
    parser.add_argument(
        "--system",
        default="proposed",
        choices=[*SYSTEMS, *ABLATIONS, "all"],
        help="b0 has no retrieval, b1 is dense retrieval, proposed verifies citations.",
    )
    parser.add_argument(
        "--papers",
        default=None,
        help="JSON corpus in the sample_papers shape. Default is the two fixture notes.",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Call Gemini. Requires GEMINI_API_KEY. Default is the offline extractive model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Abstain when the best reciprocal-rank-fusion score is below this value.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Also print a literature-matrix row for each fixture note.",
    )
    args = parser.parse_args(argv)

    try:
        model = _model(args.gemini)
    except RuntimeError as exc:
        print(exc)
        return 2

    if args.papers:
        papers = load_papers(Path(args.papers))
        print(f"corpus: {args.papers} ({len(papers)} papers)")
    else:
        papers = load_sample_papers()
        print("corpus: two local fixture notes, not the Qasper experiment set")
    index = HybridIndex.build(papers)
    print(f"generator: {model.name}")
    if model.name == "extractive":
        print("note: extractive mode quotes the top passage; it is not a Gemini answer")

    if args.system == "all":
        results = compare(args.question, index, model, threshold=args.threshold)
    elif args.system in ABLATIONS:
        results = [
            answer_ablation(
                args.question,
                index,
                model,
                args.system,
                threshold=args.threshold,
            )
        ]
    else:
        results = [
            answer(
                args.question,
                index,
                model,
                args.system,
                threshold=args.threshold,
            )
        ]
    for result in results:
        _print_result(result)

    if args.matrix:
        print("\nliterature matrix")
        for paper in papers:
            print(literature_row(paper, index, model))
            print("---")
    return 0


def _model(use_gemini: bool):
    if not use_gemini:
        return ExtractiveModel()
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key is not configured. Add GEMINI_API_KEY to the project-root .env file."
        )
    return GeminiModel(api_key)


def _print_result(result: AnswerResult) -> None:
    print()
    print(f"system: {result.system}")
    print(f"retrieval: {result.retrieval_mode}")
    print(f"abstained: {str(result.abstained).lower()}")
    if result.citation_accuracy is None:
        print("citation_accuracy: n/a")
    else:
        print(f"citation_accuracy: {result.citation_accuracy:.2f}")
    print(f"answer: {result.answer}")
    if result.citations:
        print("citations: " + ", ".join(result.citations))
    if result.dropped:
        print(f"dropped_sentences: {len(result.dropped)}")
    for decision in result.verifications:
        print(
            f"verification: {decision['label']} "
            f"kept={str(decision['kept']).lower()} "
            f"citation={decision['citation']}"
        )
    if not result.retrieved:
        print("retrieved: none")
    else:
        print("retrieved:")
        for hit in result.retrieved:
            preview = hit.chunk.text[:180].replace("\n", " ")
            print(
                f"  {hit.chunk.chunk_id}  score={hit.score:.4f}  "
                f"{hit.chunk.section}  {preview}"
            )
    if result.abstained and result.answer != ABSTAIN_MESSAGE:
        print(f"expected abstain text: {ABSTAIN_MESSAGE}")


if __name__ == "__main__":
    raise SystemExit(main())
