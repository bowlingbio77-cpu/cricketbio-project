# Source Information

## Dataset: DeepSportradar Cricket Bowl Release Challenge

### Identity
- **Name**: DeepSportradar Cricket Bowl Release Dataset
- **Year**: 2023
- **Challenge**: ACM MMSports 2023 Workshop Challenge
- **Organizer**: Sportradar (https://sportradar.com)
- **Maintainer**: Davide Zambrano (d.zambrano@sportradar.com)

### Source URLs
- **GitHub**: https://github.com/DeepSportradar/cricket-bowl-release-challenge
- **Kaggle**: https://www.kaggle.com/datasets/dzambrano/cricket-bowlrelease-dataset
- **Challenge page**: http://mmsports.multimedia-computing.de/mmsports2023/challenge.html
- **EvalAI**: https://eval.ai/web/challenges/challenge-page/2077/overview

### License
- **Dataset**: CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0)
- **Code**: Apache 2.0
- **Usage**: Research purposes only

### Dataset Statistics
- **Total videos**: 40
- **Annotated videos**: 26 (18 training + 8 testing)
- **Challenge videos**: 14 (annotations hidden)
- **Video content**: ~2 overs per video (real match footage)
- **Frame rate**: Varies by video (typically 25-30 fps)

### Annotation Methodology
- **Annotator**: Sportradar internal annotation team
- **Method**: Professional manual annotation
- **Quality**: High-quality, production-grade annotations used for commercial sports data

### Annotation Format

Each annotation file is a JSON with the following structure:

```json
{
    "event": {
        "<frame_idx>": "<event_type>"
    },
    "person": {
        "<frame_idx>": [
            {
                "person_role": "bowler|batsman|wicketkeeper|umpire|fielder",
                "bounding_box": [x1, y1, x2, y2]
            }
        ]
    }
}
```

**Event types**:
- `"is bowling"`: The bowler is in the bowling action
- `"bowl release"`: The ball is being released

**Person roles**:
- `"bowler"`: The person bowling the ball
- `"batsman"`: The person facing the delivery
- `"wicketkeeper"`: The wicketkeeper
- `"umpire"`: The umpire
- `"fielder"`: Any fielder

### Dataset Splits

**Training set** (18 videos):
```
20221001_4139131_2overs, 20221119_4139084_2overs, 20221203_4139091_2overs,
20221203_4142743_2overs, 20221203_4139088_2overs, 20221203_4139162_2overs,
20221203_4139164_2overs, 20221105_4142385_2overs, 20221203_4143496_2overs,
20221112_4142388_2overs, 20221203_4142744_2overs, 20221203_4139090_2overs,
20221112_4139153_2overs, 20220327_3916886_2overs, 20221126_4155987_2overs,
20221112_4142735_2overs, 20221112_4155986_2overs, 20221001_4139132_2overs
```

**Test set** (8 videos):
```
20221126_4139157_2overs, 20221126_4139156_2overs, 20221203_4142396_2overs,
20221105_4139079_2overs, 20221203_4139160_2overs, 20221203_4156185_2overs,
20221112_4142385_2overs, 20221001_4139134_2overs
```

### Video Naming Convention
Format: `{date}_{match_id}_2overs`
- Date: YYYYMMDD
- Match ID: Unique identifier
- Suffix: `_2overs` indicates the clip contains approximately 2 overs

### Validation Applicability

This dataset is suitable for validating:
1. **Bowler identification**: Person role labels allow verifying PaceAI selects the correct bowler
2. **Bowler tracking**: Frame-by-frame bounding boxes allow verifying tracking consistency
3. **Bowl release detection**: Event annotations provide approximate release frame ground truth
4. **Player detection**: Bounding boxes for all players allow verifying detection accuracy

**Limitations**:
- No biomechanical measurements (joint angles, etc.)
- No precise release frame (only event windows of ~100 frames)
- Broadcast camera angle (not side-on biomechanics view)
- Multiple deliveries per video (need to extract single delivery)

### Citation

If using this dataset, cite:
```
DeepSportradar Cricket Bowl Release Challenge. Sportradar. ACM MMSports 2023.
https://github.com/DeepSportradar/cricket-bowl-release-challenge
```
