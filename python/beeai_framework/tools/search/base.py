# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pydantic import BaseModel

from beeai_framework.tools import ToolOutput
from beeai_framework.utils.strings import to_json


class SearchToolResult(BaseModel):
    title: str
    description: str
    url: str


class SearchToolOutput(ToolOutput):
    def __init__(self, results: list[SearchToolResult]) -> None:
        super().__init__()
        self.results = results

    def get_text_content(self) -> str:
        if not self.results:
            return "No results were found for a given query"

        return to_json(self.results)

    def is_empty(self) -> bool:
        return len(self.results) == 0

    def sources(self) -> list[str]:
        return [result.url for result in self.results]
