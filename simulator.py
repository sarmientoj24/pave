import json
import requests
import fire


def request(filename, port=3000):
    # Create request
    data = {"filename": filename}

    result = requests.post(
        f"http://0.0.0.0:{port}/invocation",
        headers={"content-type": "application/json"},
        data=json.dumps(data),
    )

    txt_results = result.text
    print(txt_results)


if __name__ == "__main__":
    fire.Fire(request)
