"""Current user-authorized comparison scope and explicit budget interpretation."""
TRAIN_NAMES=['rgb3_short90','rgb4','rgb3_event4','event3_rgb4','event3_event4','rgb3_event3','rgb4_event4']
PRETRAIN_NAMES=['pre_rgb3','pre_rgb4']
EVAL_NAMES=PRETRAIN_NAMES+TRAIN_NAMES
HIST_NAMES=['rgb3','rgb4','rgb3_event4','event3_rgb4','event3_event4']
FUSION_NAMES=[n for n in TRAIN_NAMES if n not in ('rgb3_short90','rgb4')]
LABELS=dict(rgb3='device3 RGB',rgb4='device4 RGB',rgb3_event4='device3 RGB + device4 event',
    event3_rgb4='device3 event + device4 RGB',event3_event4='device3 event + device4 event',
    rgb3_event3='device3 RGB + device3 event',rgb4_event4='device4 RGB + device4 event',
    pre_rgb3='원본 device3 RGB',pre_rgb4='원본 device4 RGB',rgb3_short90='device3 RGB')
PAIRS=[('rgb3_short90','rgb3_event4'),('rgb4','event3_rgb4'),('rgb3_short90','event3_event4'),('rgb4','event3_event4'),
    ('old_rgb3_event4','rgb3_event4'),('old_event3_rgb4','event3_rgb4'),('old_event3_event4','event3_event4'),
    ('old_rgb3','rgb3_short90'),('old_rgb4','rgb4'),('rgb3_short90','rgb3_event3'),('rgb4','rgb4_event4'),
    ('pre_rgb3','rgb3_short90'),('pre_rgb4','rgb4')]
PRIMARY_POPULATION='common9'
HISTORICAL_POPULATION='common14'
