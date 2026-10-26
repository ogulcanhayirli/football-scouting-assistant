"""The Ask tab's chat (prompt 20, decision 11): a prototype that answers a
fan's questions from the published tables only.

One AWS Lambda behind a Function URL (``infra/scout_infra/chat_stack.py``)
runs ``handler.handler``. At cold start it reads the files ``scout publish``
put in the site bucket (``data``), and every answer is built by Claude
Sonnet 5.5 from four tools over them (``tools``), under the rules of the
system prompt (``prompt``). ``guard`` holds the caps (20 messages an hour per
visitor, 1 USD a day, 5 USD a month) in a small S3 object, and ``agent`` runs
the tool loop (at most six tool calls a turn) and prices every reply from
its usage fields.

This package needs only the standard library, ``anthropic`` and (in the
Lambda) ``boto3``: no pandas, so it fits a small bundle. Tests use a fake
client and a local folder of the published files.
"""
