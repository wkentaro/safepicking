import message_filters


class MessageSubscriber:
    def __init__(self, topics):
        self._topics = topics  # [(topic_name, topic_msg), ...]
        self._subscribers = []
        self.msgs = None

    def subscribe(self):
        subscribers = []
        for topic_name, topic_msg in self._topics:
            sub = message_filters.Subscriber(topic_name, topic_msg)
            subscribers.append(sub)
        self._subscribers = subscribers

        sync = message_filters.TimeSynchronizer(
            self._subscribers,
            queue_size=50,
        )
        sync.registerCallback(self._callback)

    def unsubscribe(self):
        for sub in self._subscribers:
            sub.unregister()

    def _callback(self, *msgs):
        self.msgs = msgs
