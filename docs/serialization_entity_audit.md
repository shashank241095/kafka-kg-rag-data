# Serialization audit of the most frequent extracted phrases

Source of truth: `artifacts/hipporag/frozen_corpus_1600.jsonl` (SHA256
`950cf89775df0f4d2a6b5940c0d1246bd1655315481dd4322b12825885351889`, mode 444). The corpus
was not modified. Entity-occurrence counts come from
`work/output/openie_kafka_ast_results_ner_gpt-3.5-turbo-1106_1600.json`.

## The frozen serialization

Every one of the 1,600 records is rendered by
`scripts/embed_logs_to_qdrant.py::build_point_text` as:

```
[{LEVEL}] {template} (logger={logger} method={method} class={class} location={file}:{line})
```

So `logger=`, `method=`, `class=` and `location=` are literal field keys present in
all 1,600 records, and the bracketed leading token is the *value* of the level field.

## Classification

A = literal serialization field/key · B = frequent value of a serialization field ·
C = generic logging vocabulary from record content · D = domain-semantic content · E = ambiguous

| Phrase | Extracted occurrences | Field key present | Level-field value | In message template | Class |
|---|---|---|---|---|---|
| `logger` | 379 | 1600/1600 | 0/1600 | 0/1600 | **A** |
| `debug` | 307 | 0/1600 | 567/1600 | 0/1600 | **B** |
| `error` | 263 | 0/1600 | 272/1600 | 249/1600 | **B + C** |
| `location` | 250 | 1600/1600 | 0/1600 | 0/1600 | **A** |
| `class` | 210 | 1600/1600 | 0/1600 | 0/1600 | **A** |

### `logger` — classification A

A literal field key in 1,600/1,600 records. It is never a level value and never appears
in message text. It does occur inside 37 logger *names* (e.g. loggers whose class path
contains the token), so it is not exclusively a key, but its presence is dominated by the
schema. Representative records:

- `[WARN] Couldn't resolve server {} from {} as DNS resolution of the canonical hostname {} failed for {} (logger=log method=org.apache.kafka.clients.Cli…`
- `[WARN] Couldn't resolve server {} from {} as DNS resolution failed for {} (logger=log method=org.apache.kafka.clients.ClientUtils#parseAndValidateAddr…`

### `class` — classification A

A literal field key in 1,600/1,600 records; never a level value; never in message text.

- `…class=org.apache.kafka.clients.ClientUtils location=clients/src/main/java/org/apache/kafka/clie…`
- `…class=org.apache.kafka.clients.ClientUtils location=clients/src/main/java/org/apache/kafka/clie…`

### `location` — classification A

A literal field key in 1,600/1,600 records; never a level value; never in message text.

- `…location=clients/src/main/java/org/apache/kafka/clients/ClientUtils.java:82)…`
- `…location=clients/src/main/java/org/apache/kafka/clients/ClientUtils.java:90)…`

### `debug` — classification B

Not a field key. It is the *value* of the level field in 567/1,600 records and appears in
no message template. Representative records:

- `[DEBUG] Resolved host {} as {} (logger=log method=org.apache.kafka.clients.ClientUtils#resolve(String, HostResolver) class=org.apache.kafka.clien…`
- `[DEBUG] Resolved host {} to addresses {} (logger=log method=org.apache.kafka.clients.ClusterConnectionStates.NodeConnectionState#resolveAddresses…`

### `error` — classification B + C

Not a field key. It is the level value in 272/1,600 records **and** independently appears
in the message text of 249/1,600 records, so it is partly a schema value and partly
genuine logging vocabulary carried by record content. Representative records:

- level value: `[ERROR] Metadata response reported invalid topics {} (logger=log method=org.apache.kafka.clients.Metadata#checkInvalidTopics(Cluster) class=…`
- in message text: `[INFO] Error sending fetch request {} to node {}: (logger=log method=org.apache.kafka.clients.FetchSessionHandler#handleError(Throwable) cla…`
- in message text: `[DEBUG] Requesting metadata update for partition {} due to error {} (logger=log method=org.apache.kafka.clients.Metadata#handleMetadataRespo…`

## Level-field value distribution

| Level | Records |
|---|---|
| DEBUG | 567 |
| INFO | 368 |
| ERROR | 272 |
| TRACE | 196 |
| WARN | 190 |
| FATAL | 7 |

## Conclusion

Three of the five most frequent extracted phrases (`logger`, `class`, `location`) are
literal serialization field keys present in every record. One (`debug`) is a frequent
value of the level field. One (`error`) is both a level value and generic logging
vocabulary occurring in message content. **None of the five is a Kafka-specific domain
entity**, but it is *not* accurate to say all five are field labels: two are field values,
and `error` is partly content-bearing.

Supported statement: the most frequent extracted phrases were dominated by record-schema
metadata and generic logging vocabulary rather than Kafka-specific entities. Not
supported: that OpenIE extracted only the schema — domain entities such as
`topicpartition` (197), `partition` (140), `node` (123) and `kafka` (121) also occur
frequently, and the extraction produced 2,787 unique entity strings overall.
