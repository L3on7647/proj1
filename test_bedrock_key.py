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


def main():
    print("=" * 60)
    print("AWS Bedrock API Key Test")
    print("=" * 60)

    key_id, key_secret = decode_key(ENCODED_KEY)

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
