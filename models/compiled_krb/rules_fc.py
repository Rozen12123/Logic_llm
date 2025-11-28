# rules_fc.py

from pyke import contexts, pattern, fc_rule, knowledge_base

pyke_version = '1.1.1'
compiler_version = 1

def fact1(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Impus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Transparent',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact2(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Impus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Tumpus',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact3(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Tumpus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Angry',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact4(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Tumpus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Dumpus',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact5(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Dumpus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Orange',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact6(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Jompus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Dumpus',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact7(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Jompus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Feisty',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact8(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Jompus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Numpus',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact9(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Numpus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Earthy',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact10(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Rompus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Earthy',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact11(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Numpus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Vumpus',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact12(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Vumpus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Temperate',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact13(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Vumpus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Wumpus',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact14(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Wumpus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Small',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact15(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Wumpus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Yumpus',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact16(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Yumpus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Metallic',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact17(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Yumpus', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Zumpus',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def populate(engine):
  This_rule_base = engine.get_create('rules')
  
  fc_rule.fc_rule('fact1', This_rule_base, fact1,
    (('facts', 'Impus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact2', This_rule_base, fact2,
    (('facts', 'Impus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact3', This_rule_base, fact3,
    (('facts', 'Tumpus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact4', This_rule_base, fact4,
    (('facts', 'Tumpus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact5', This_rule_base, fact5,
    (('facts', 'Dumpus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(False),))
  
  fc_rule.fc_rule('fact6', This_rule_base, fact6,
    (('facts', 'Jompus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact7', This_rule_base, fact7,
    (('facts', 'Jompus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact8', This_rule_base, fact8,
    (('facts', 'Jompus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact9', This_rule_base, fact9,
    (('facts', 'Numpus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(False),))
  
  fc_rule.fc_rule('fact10', This_rule_base, fact10,
    (('facts', 'Rompus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact11', This_rule_base, fact11,
    (('facts', 'Numpus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact12', This_rule_base, fact12,
    (('facts', 'Vumpus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact13', This_rule_base, fact13,
    (('facts', 'Vumpus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact14', This_rule_base, fact14,
    (('facts', 'Wumpus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact15', This_rule_base, fact15,
    (('facts', 'Wumpus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact16', This_rule_base, fact16,
    (('facts', 'Yumpus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(False),))
  
  fc_rule.fc_rule('fact17', This_rule_base, fact17,
    (('facts', 'Yumpus',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))


Krb_filename = '../symbolic_solvers/pyke_solver/.cache_program/rules.krb'
Krb_lineno_map = (
    ((12, 16), (3, 3)),
    ((17, 19), (5, 5)),
    ((28, 32), (9, 9)),
    ((33, 35), (11, 11)),
    ((44, 48), (15, 15)),
    ((49, 51), (17, 17)),
    ((60, 64), (21, 21)),
    ((65, 67), (23, 23)),
    ((76, 80), (27, 27)),
    ((81, 83), (29, 29)),
    ((92, 96), (33, 33)),
    ((97, 99), (35, 35)),
    ((108, 112), (39, 39)),
    ((113, 115), (41, 41)),
    ((124, 128), (45, 45)),
    ((129, 131), (47, 47)),
    ((140, 144), (51, 51)),
    ((145, 147), (53, 53)),
    ((156, 160), (57, 57)),
    ((161, 163), (59, 59)),
    ((172, 176), (63, 63)),
    ((177, 179), (65, 65)),
    ((188, 192), (69, 69)),
    ((193, 195), (71, 71)),
    ((204, 208), (75, 75)),
    ((209, 211), (77, 77)),
    ((220, 224), (81, 81)),
    ((225, 227), (83, 83)),
    ((236, 240), (87, 87)),
    ((241, 243), (89, 89)),
    ((252, 256), (93, 93)),
    ((257, 259), (95, 95)),
    ((268, 272), (99, 99)),
    ((273, 275), (101, 101)),
)
