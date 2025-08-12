from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
import torch
from sentence_transformers import SentenceTransformer


def cosine_similarity(embeddings1, embeddings2):
    """
    Args:
        embeddings1: an embedding vector in torch.Tensor format.
        embeddings2: another embedding vector in torch.Tensor format.
    """
    dot_product = torch.sum(embeddings1 * embeddings2)
    norm1 = torch.norm(embeddings1)
    norm2 = torch.norm(embeddings2)
    similarity = dot_product / (norm1 * norm2)
    return similarity.item()


def main():
    prompts = [
        "The capital of France is",
    ]
    sbert_model_path = "/mnt/task_runtime/jp_bert/model/maps-jb-bert-pca/"
    sbert = SentenceTransformer(sbert_model_path)
    sbert_embeddings = sbert.encode(prompts)
    print(sbert_embeddings)
    print(f"shape: {sbert_embeddings.shape}")

    print("=======")

    sampling_params = SamplingParams(max_tokens=32, return_context_logits=True)
    llm = LLM(model='/mnt/task_runtime/jp_bert/converted-classification-model/')
    outputs = llm.generate(prompts, sampling_params)

    tllm_logits = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        tllm_logit = output.context_logits.cpu()[0, :]
        print(f"Prompt: {prompt!r}, Context logits: {tllm_logit}")
        tllm_logits += [tllm_logit]
    # Stack the output
    tllm_logits = torch.stack(tllm_logits)
    print(tllm_logits)
    print(f"shape: {tllm_logits.shape}")

    # cosine similarity
    simi = cosine_similarity(torch.tensor(sbert_embeddings[0]), tllm_logits[0])
    print(simi)


if __name__ == '__main__':
    main()
