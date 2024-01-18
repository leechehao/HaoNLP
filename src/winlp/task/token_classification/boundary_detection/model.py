import transformers

from winlp.task.token_classification import TokenClassificationModule


class BoundaryDetectionModule(TokenClassificationModule):

    @property
    def hf_pipeline_task(self):
        aggregation_strategy = None
        return transformers.TokenClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            aggregation_strategy=aggregation_strategy,
        )
