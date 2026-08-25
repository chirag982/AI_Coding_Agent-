from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
import subprocess

load_dotenv()

client = InferenceClient(
    api_key=os.environ.get("API_token"),
    model="google/gemma-4-31B-it"
)

WORKSPACE = "workspace"
SOLUTION_PATH = f"{WORKSPACE}/solution.py"
TEST_PATH = f"{WORKSPACE}/test_solution.py"


# =========================================================
# THE HANDS — tools the agent can call
# =========================================================

def write_file(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}"


def run_tests(path: str) -> str:
    result = subprocess.run(
        ["python", "-m", "pytest", path, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    summary_line = ""
    for line in output.splitlines():
        if ("passed" in line or "failed" in line or "error" in line) and "==" in line:
            summary_line = line.strip("= ").strip()

    status = "FAILED" if ("failed" in output.lower() or "error" in output.lower()) else "PASSED"

    return f"STATUS: {status}\nSUMMARY: {summary_line}\n\nFULL OUTPUT:\n{output}"


TOOL_REGISTRY = {
    "write_file": write_file,
    "run_tests": run_tests,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Save code content to a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write to, e.g. 'workspace/solution.py'"},
                    "content": {"type": "string", "description": "Full code to write, no markdown fences."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run pytest against the given test file and return pass/fail results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the test file, e.g. 'workspace/test_solution.py'"}
                },
                "required": ["path"],
            },
        },
    },
]


# =========================================================
# THE BRAIN + EYES + LOOP
# =========================================================

class Agent:
    def __init__(self, client, system: str = "", tools: list = None, max_iterations: int = 8) -> None:
        self.client = client
        self.messages: list = []
        self.tools = tools if tools is not None else []
        self.max_iterations = max_iterations
        if system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message: str = ""):
        if message:
            self.messages.append({"role": "user", "content": message})
        return self.execute()

    def execute(self):
        iterations = 0

        while True:
            iterations += 1
            if iterations > self.max_iterations:
                return f"Stopped: hit max iterations ({self.max_iterations}) without passing tests."

            completion = self.client.chat.completions.create(
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
            )
            response_message = completion.choices[0].message

            if response_message.tool_calls:
                self.messages.append(response_message)
                tool_outputs = []

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    function_to_call = TOOL_REGISTRY.get(function_name)

                    if function_to_call:
                        executed_output = function_to_call(**function_args)
                        tool_output_content = str(executed_output)
                    else:
                        tool_output_content = f"Error: tool '{function_name}' not found"

                    print(f"[tool call] {function_name}({function_args})\n -> {tool_output_content[:300]}\n")

                    # Stop early if tests already passed — no need to wait for the model to notice
                    if function_name == "run_tests" and "STATUS: PASSED" in tool_output_content:
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": tool_output_content + "\n\nAll tests passed. You are done.",
                        })
                        self.messages.extend(tool_outputs)
                        return "Done — all tests passed."

                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_output_content,
                    })

                self.messages.extend(tool_outputs)

            else:
                return response_message.content


# =========================================================
# RUN IT
# =========================================================

if __name__ == "__main__":
    agent = Agent(
        client=client,
        system=(
            "You are a coding agent. Your job: write correct Python code that "
            f"passes the tests in {TEST_PATH}. "
            f"Always use write_file to save code to {SOLUTION_PATH}, then use "
            f"run_tests on {TEST_PATH} to check it. "
            "Never output code as plain text — always save it with write_file first. "
            "If tests fail, read the failure output, fix your code, write_file again, "
            "and run_tests again. Keep going until all tests pass."
        ),
        tools=TOOL_SCHEMAS,
        max_iterations=8,
    )

    task = input("Describe the coding task -> ")

    goal = (
        f"Task: {task}\n"
        f"A test file already exists at {TEST_PATH}. Write your solution to "
        f"{SOLUTION_PATH} so those tests pass. Use write_file then run_tests, "
        f"and iterate until everything passes."
    )

    result = agent(goal)
    print("\n--- FINAL RESULT ---")
    print(result)