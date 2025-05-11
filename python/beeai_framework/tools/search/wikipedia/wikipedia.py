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


try:
    import wikipediaapi  # type: ignore
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [wikipedia] not found.\nRun 'pip install \"beeai-framework[wikipedia]\"' to install."
    ) from e

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.search import SearchToolOutput, SearchToolResult
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions


class WikipediaToolInput(BaseModel):
    query: str = Field(description="Search query, name of the Wikipedia page.")
    summary_only: bool = Field(
        description="If set to true, only page's summary will be provided instead of a full text.",
        default=False,
    )
    # section_titles: bool = Field(
    #    description="If set to true, it returns section titles as the description.", default=False
    # )
    language: str | None = Field(description="Retrieves the specified language version if available.", default=None)


class WikipediaToolResult(SearchToolResult):
    pass


class WikipediaToolOutput(SearchToolOutput):
    pass


class WikipediaTool(Tool[WikipediaToolInput, ToolRunOptions, WikipediaToolOutput]):
    name = "Wikipedia"
    description = "Search factual and historical information, including biography, \
        history, politics, geography, society, culture, science, technology, people, \
        animal species, mathematics, and other subjects."
    input_schema = WikipediaToolInput
    client = wikipediaapi.Wikipedia(
        user_agent="beeai-framework https://github.com/i-am-bee/beeai-framework", language="en"
    )

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "search", "wikipedia"],
            creator=self,
        )

    def get_section_titles(self, sections: wikipediaapi.WikipediaPage.sections) -> str:
        titles = []
        for section in sections:
            titles.append(section.title)
        return ",".join(str(title) for title in titles)

    async def _run(
        self, input: WikipediaToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> WikipediaToolOutput:
        page_py = self.client.page(input.query)

        if not page_py.exists():
            return WikipediaToolOutput([])

        if input.language is not None and input.language in page_py.langlinks:
            page_py = page_py.langlinks[input.language]

        description_output = page_py.text if not input.summary_only else page_py.summary

        return WikipediaToolOutput(
            [
                WikipediaToolResult(
                    title=page_py.title or input.query,
                    description=description_output or "",
                    url=page_py.fullurl or "",
                )
            ]
        )
