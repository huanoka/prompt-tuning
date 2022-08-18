from transformers import T5Model
from openprompt import PromptForGeneration


class MobilePromptForGeneration(PromptForGeneration):
    def forward(self, *args, **kwargs):

