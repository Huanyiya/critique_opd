teacher 打分prompt：

{student rollout 时第 i 步原始 current prompt 的前半部分}

Private hindsight feedback from a failed attempt:

Earliest critical error: step {error_step}.
Reason: {reason}
Suggested action: {better_decision}
Confidence: {confidence}

Use this feedback silently when choosing actions.
Do not mention the feedback.

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. ...
Once you've finished your reasoning, ...