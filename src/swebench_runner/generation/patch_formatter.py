"""Format patches for evaluation compatibility."""


class PatchFormatter:
    """Ensures patches are in the correct format for evaluation."""

    @staticmethod
    def format_for_evaluation(
        generation_results: list
    ) -> list:
        """Convert generation format to evaluation format.

        The evaluation expects:
        - 'instance_id': str
        - 'patch': str (the actual diff)
        - No 'prediction' field (common mistake)
        """
        formatted = []
        for result in generation_results:
            # Ensure we have required fields
            if 'instance_id' not in result or 'patch' not in result:
                continue

            formatted_entry = {
                'instance_id': result['instance_id'],
                'patch': result['patch']
            }

            # Optionally include model info as metadata
            if 'model' in result:
                formatted_entry['model'] = result['model']

            formatted.append(formatted_entry)

        return formatted
