from flask import Flask, request, jsonify
import os

app = Flask(__name__)
dim = int(os.getenv("MEMORY_EMBEDDING_DIMENSION", "8"))

@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    body = request.get_json()
    inputs = body.get("input", [])
    data = []
    for i, _ in enumerate(inputs):
        vec = [float((i + 1) % (j + 1)) for j in range(dim)]
        data.append({"index": i, "embedding": vec})
    return jsonify({"data": data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
