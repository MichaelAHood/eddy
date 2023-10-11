import uuid
import random
from datetime import datetime, timedelta


def generate_messages(num_uids):
    """Create fake messages metadata."""

    distinct_ids = [str(uuid.uuid4()) for _ in range(num_uids)]

    messages = []

    # Generate 3 to 10 messages for each UUID
    for id in distinct_ids:
        num_messages = random.randint(3, 10)
        for _ in range(num_messages):
            from_id = id
            to_id = random.choice(
                [x for x in distinct_ids if x != id]
            )  # ensure 'from' and 'to' are different
            start = datetime.now()  # current timestamp as an example
            end = start + timedelta(seconds=random.randint(10, 120))

            message = {
                "from": from_id,
                "to": to_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
            }

            messages.append(message)

    return messages
