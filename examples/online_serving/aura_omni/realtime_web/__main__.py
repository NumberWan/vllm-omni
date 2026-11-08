# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

from examples.online_serving.aura_omni.realtime_web.tools import AuraToolExecutor
from examples.online_serving.realtime_web.server import main

if __name__ == "__main__":
    # AURA_DEMO_TOOLS=0 serves the page without function calling.
    tools_enabled = os.environ.get("AURA_DEMO_TOOLS", "1").strip().lower() not in {"0", "false", "off", "no"}
    main(default_profile="aura-ptt", tool_executor=AuraToolExecutor() if tools_enabled else None)
