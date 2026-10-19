from app.rag.configs.cancer_pipeline import build_cancer_pipeline
from app.rag import RAGContext

p = build_cancer_pipeline(critic_mode="llm")
ctx = RAGContext(
    question="What is the mechanism of trastuzumab in HER2-positive breast cancer?",
    session_id="diag",
)
ctx = p.run(ctx)

print()
print("=== RETRY ===")
print("retry_count:", ctx.retry_count)
print("grounded:", ctx.grounded)
print("grounded_reason:", ctx.grounded_reason)

print()
print("=== ANSWER ===")
print(ctx.answer)

print()
print("=== CONTEXT (first 4000 chars) ===")
print(ctx.metadata.get("context_string", "")[:4000])