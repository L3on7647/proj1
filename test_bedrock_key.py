"""
AWS Bedrock API key test script.

Decodes the base64-encoded Bedrock API key and tests it against
the Claude model via multiple authentication approaches.
"""

import base64
import json
import sys


ENCODED_KEY = (
    "ABSKQmVkcm9ja0FQSUtleS05OWYyLWF0LTczNTg3MTg2MDQ5MDpGTnV1ajBsL3BxSFl5R2sw"
    "cGNiQ3VsYlE5cE9iYms5dlpJeDluekZxTUJrRXlvb01YSkVoQkNicDJwdz0="
)

TEST_PROMPT = "Say 'API key works!' and nothing else."


def decode_key(encoded: str) -> tuple[str, str]:
    """Decode base64 key and return (key_id, key_secret)."""
    raw = base64.b64decode(encoded)
    # First 3 bytes are a binary header; the rest is the text key
    text = raw[3:].decode("utf-8")
    key_id, key_secret = text.split(":", 1)
    return key_id, key_secret


def test_direct_anthropic(api_key: str) -> dict:
    """Test using the full decoded key as a direct Anthropic API key."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=50,
            messages=[{"role": "user", "content": TEST_PROMPT}],
        )
        text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        return {"success": True, "response": text, "model": response.model}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def test_bedrock_boto3(key_id: str, key_secret: str, region: str = "us-east-1") -> dict:
    """Test using the key parts as AWS credentials with boto3 Bedrock runtime."""
    try:
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=key_id,
            aws_secret_access_key=key_secret,
        )

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": TEST_PROMPT}],
        })

        response = client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]
        return {"success": True, "response": text, "model": result.get("model")}
    except ImportError:
        return {"success": False, "error": "boto3 not installed"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def test_anthropic_bedrock_sdk(key_id: str, key_secret: str, region: str = "us-east-1") -> dict:
    """Test using AnthropicBedrock SDK with key parts as AWS credentials."""
    try:
        import os
        import anthropic

        os.environ["AWS_ACCESS_KEY_ID"] = key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = key_secret
        os.environ["AWS_DEFAULT_REGION"] = region

        client = anthropic.AnthropicBedrock(aws_region=region)
        response = client.messages.create(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            max_tokens=50,
            messages=[{"role": "user", "content": TEST_PROMPT}],
        )
        text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        return {"success": True, "response": text, "model": response.model}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    finally:
        for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"):
            os.environ.pop(var, None)


def test_bearer_token_http(bearer_token: str, region: str = "us-east-1") -> dict:
    """Test using AWS_BEARER_TOKEN_BEDROCK as HTTP Bearer token against Bedrock runtime."""
    try:
        import urllib.request

        url = (
            f"https://bedrock-runtime.{region}.amazonaws.com"
            "/model/anthropic.claude-3-5-sonnet-20241022-v2:0/invoke"
        )
        payload = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": TEST_PROMPT}],
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {bearer_token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        text = result["content"][0]["text"]
        return {"success": True, "response": text, "model": result.get("model")}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def test_bearer_token_boto3(bearer_token: str, region: str = "us-east-1") -> dict:
    """Test AWS_BEARER_TOKEN_BEDROCK style auth via boto3 unsigned + Bearer header injection."""
    try:
        import boto3
        import botocore
        import botocore.config

        session = boto3.Session(region_name=region)

        # First verify auth by listing models (lightweight call)
        bedrock = session.client(
            "bedrock",
            region_name=region,
            config=botocore.config.Config(signature_version=botocore.UNSIGNED),
        )

        def add_bearer(request, **kwargs):
            request.headers["Authorization"] = f"Bearer {bearer_token}"

        bedrock.meta.events.register("before-send.bedrock.ListFoundationModels", add_bearer)
        resp = bedrock.list_foundation_models(byProvider="Anthropic")
        active = [
            m["modelId"] for m in resp.get("modelSummaries", [])
            if m.get("modelLifecycle", {}).get("status") == "ACTIVE"
        ]

        # Now try invoking with inference profile IDs
        client = session.client(
            "bedrock-runtime",
            region_name=region,
            config=botocore.config.Config(signature_version=botocore.UNSIGNED),
        )
        client.meta.events.register("before-send.bedrock-runtime.InvokeModel", add_bearer)

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": TEST_PROMPT}],
        })

        # Try cross-region inference profiles first (us. prefix)
        candidates = [f"us.{m}" for m in active] + active
        for model_id in candidates:
            try:
                response = client.invoke_model(
                    modelId=model_id, body=body,
                    contentType="application/json", accept="application/json",
                )
                result = json.loads(response["body"].read())
                text = result["content"][0]["text"]
                return {"success": True, "response": text, "model": model_id}
            except Exception as exc:
                err = str(exc)
                if "ThrottlingException" in err and "Too many tokens per day" in err:
                    # Auth succeeded — quota exhausted
                    return {
                        "success": True,
                        "response": "(daily token quota exhausted — auth succeeded)",
                        "model": model_id,
                        "note": "ThrottlingException after auth",
                    }
                # continue to next model on other errors

        return {"success": False, "error": "No model responded successfully"}
    except ImportError:
        return {"success": False, "error": "boto3 not installed"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def main():
    print("=" * 60)
    print("AWS Bedrock API Key Test")
    print("=" * 60)

    key_id, key_secret = decode_key(ENCODED_KEY)
    bearer_token = ENCODED_KEY  # raw base64 token as-is

    print(f"\nKey ID    : {key_id}")
    print(f"Key Secret: {key_secret[:8]}...{key_secret[-4:]} (masked)")
    print()

    results = {}

    # Test 1: full decoded string as direct Anthropic API key
    full_key = f"{key_id}:{key_secret}"
    print("Test 1: Direct Anthropic SDK (full key as api_key)...")
    results["direct_anthropic"] = test_direct_anthropic(full_key)
    r = results["direct_anthropic"]
    if r["success"]:
        print(f"  ✓ SUCCESS — model: {r['model']}, response: {r['response']}")
    else:
        print(f"  ✗ FAILED  — {r['error'][:120]}")

    # Test 2: boto3 Bedrock runtime with key parts as AWS credentials
    print("\nTest 2: boto3 bedrock-runtime (key_id as AWS access key)...")
    results["boto3_bedrock"] = test_bedrock_boto3(key_id, key_secret)
    r = results["boto3_bedrock"]
    if r["success"]:
        print(f"  ✓ SUCCESS — model: {r['model']}, response: {r['response']}")
    else:
        print(f"  ✗ FAILED  — {r['error'][:120]}")

    # Test 3: Anthropic Bedrock SDK with key parts as AWS env vars
    print("\nTest 3: AnthropicBedrock SDK (key_id as AWS_ACCESS_KEY_ID)...")
    results["anthropic_bedrock_sdk"] = test_anthropic_bedrock_sdk(key_id, key_secret)
    r = results["anthropic_bedrock_sdk"]
    if r["success"]:
        print(f"  ✓ SUCCESS — model: {r['model']}, response: {r['response']}")
    else:
        print(f"  ✗ FAILED  — {r['error'][:120]}")

    # Test 4: raw HTTP Bearer token (AWS_BEARER_TOKEN_BEDROCK style)
    print("\nTest 4: HTTP Bearer token (AWS_BEARER_TOKEN_BEDROCK)...")
    results["bearer_http"] = test_bearer_token_http(bearer_token)
    r = results["bearer_http"]
    if r["success"]:
        print(f"  ✓ SUCCESS — model: {r['model']}, response: {r['response']}")
    else:
        print(f"  ✗ FAILED  — {r['error'][:120]}")

    # Test 5: boto3 with unsigned sig + Bearer header injection
    print("\nTest 5: boto3 unsigned + Bearer header injection...")
    results["bearer_boto3"] = test_bearer_token_boto3(bearer_token)
    r = results["bearer_boto3"]
    if r["success"]:
        print(f"  ✓ SUCCESS — model: {r['model']}, response: {r['response']}")
    else:
        print(f"  ✗ FAILED  — {r['error'][:120]}")

    print("\n" + "=" * 60)
    successful = [name for name, r in results.items() if r["success"]]
    if successful:
        print(f"✓ Key is VALID — working method(s): {', '.join(successful)}")
        return 0
    else:
        print("✗ Key did NOT authenticate with any tested method.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
