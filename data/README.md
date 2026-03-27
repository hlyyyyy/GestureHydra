# Data Placeholder

This directory is intentionally kept out of version control except for this document.

Suggested layout for the local Streamer dataset:

```text
data/streamer/
|-- audio/                    # wav or other decoded audio files
|-- motion/                   # motion sequences, joints, or body parameters
|-- text/                     # ASR transcripts or aligned text
|-- annotations/
|   |-- train.json
|   |-- val.json
|   `-- test.json
`-- retrieval_repository/     # semantic gesture clips used by RAG
```

Suggested metadata fields per sample:

- `sample_id`
- `speaker_id`
- `audio_path`
- `motion_path`
- `text_path`
- `num_frames`
- `fps`
- `audio_sample_rate`
- `semantic_labels`
- `start_time`
- `end_time`

You can adapt this structure once the final preprocessing pipeline is ready.
