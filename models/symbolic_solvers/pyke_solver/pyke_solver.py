import os
import re
import shutil
from pathlib import Path
from pyke import knowledge_engine

class Pyke_Program:
    def __init__(self, logic_program:str, dataset_name = 'ProntoQA') -> None:
        self.logic_program = logic_program
        self.flag = self.parse_logic_program()
        self.dataset_name = dataset_name
        file_path = Path(__file__).resolve()
        parent_paths = list(file_path.parents)
        compiled_candidates = []
        if len(parent_paths) > 2:
            compiled_candidates.append(parent_paths[2] / 'compiled_krb')  # .../models/compiled_krb
        if len(parent_paths) > 3:
            compiled_candidates.append(parent_paths[3] / 'compiled_krb')  # project root / compiled_krb
        seen = set()
        self.compiled_dirs = []
        for candidate in compiled_candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            self.compiled_dirs.append(resolved)
            seen.add(resolved)
        
        # create the folder to save the Pyke program
        cache_dir = os.path.join(os.path.dirname(__file__), '.cache_program')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

        # prepare the files for facts and rules
        try:
            self.create_fact_file(self.Facts)
            self.create_rule_file(self.Rules)
            self.flag = True
        except:
            self.flag = False

        self.answer_map = {'ProntoQA': self.answer_map_prontoqa, 
                           'ProofWriter': self.answer_map_proofwriter}

    def clean_logic_program(self, program_str):
        """Clean the logic program by removing markdown formatting and extracting key sections"""
        keywords = ['Query:', 'Rules:', 'Facts:', 'Predicates:']
        
        # First, try to find and convert alternative formats used by some models
        # Pattern 1: "**1) Define all the predicates**" -> "Predicates:"
        program_str = re.sub(r'\*\*\d+\)\s*[Dd]efine all the predicates.*?\*\*', 'Predicates:', program_str, flags=re.IGNORECASE)
        # Pattern 2: "**2) Parse the problem into logic rules**" -> "Rules:"
        program_str = re.sub(r'\*\*\d+\)\s*[Pp]arse the problem into logic rules.*?\*\*', 'Rules:', program_str, flags=re.IGNORECASE)
        # Pattern 3: "**3) Write all the facts**" -> "Facts:"
        program_str = re.sub(r'\*\*\d+\)\s*[Ww]rite all the facts.*?\*\*', 'Facts:', program_str, flags=re.IGNORECASE)
        # Pattern 4: "**4) Parse the question**" or "Query:" -> "Query:"
        program_str = re.sub(r'\*\*\d+\)\s*[Pp]arse the question.*?\*\*', 'Query:', program_str, flags=re.IGNORECASE)
        
        # Normalize keyword formats (remove markdown formatting from keywords)
        for keyword in keywords:
            # Remove markdown bold/italic from keywords: **Query:** or *Query:* -> Query:
            program_str = re.sub(rf'\*\*{re.escape(keyword)}\*\*', keyword, program_str, flags=re.IGNORECASE)
            program_str = re.sub(rf'\*{re.escape(keyword)}\*', keyword, program_str, flags=re.IGNORECASE)
            # Handle markdown headers: # Query: -> Query:
            program_str = re.sub(rf'#+\s*{re.escape(keyword)}', keyword, program_str, flags=re.IGNORECASE)
        
        # Extract sections starting from the first keyword found
        # Find positions of all keywords
        keyword_positions = []
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            match = pattern.search(program_str)
            if match:
                keyword_positions.append((match.start(), keyword))
        
        if keyword_positions:
            # Sort by position
            keyword_positions.sort(key=lambda x: x[0])
            # Extract from the first keyword to the end
            first_keyword_pos = keyword_positions[0][0]
            program_str = program_str[first_keyword_pos:]
        else:
            # If no standard keywords found, try to extract from alternative patterns
            # Look for "Query:" at the end (common pattern)
            query_match = re.search(r'Query:\s*([^\n]+)', program_str, re.IGNORECASE)
            if query_match:
                # Try to reconstruct from the structure
                # Extract predicates (lines starting with * and containing ($x, bool))
                predicates = re.findall(r'\*\s+\*\*?(\w+\(\$x,\s*bool\))\*\*?.*?:::', program_str)
                # Extract rules (lines containing >>>)
                rules = re.findall(r'\*\s+([^\n]*>>>[^\n]*)', program_str)
                # Extract facts (lines in section 3)
                facts = re.findall(r'\*\s+([A-Z]\w+\([^)]+\))', program_str)
                
                if predicates or rules or facts:
                    # Reconstruct in standard format
                    result = []
                    if predicates:
                        result.append("Predicates:")
                        result.extend([p + " :::" for p in predicates])
                    if facts:
                        result.append("Facts:")
                        result.extend(facts)
                    if rules:
                        result.append("Rules:")
                        result.extend(rules)
                    if query_match:
                        result.append("Query:")
                        result.append(query_match.group(1).strip())
                    program_str = '\n'.join(result)
        
        # Remove markdown formatting from content (but keep the structure)
        # Remove bold: **text** -> text
        program_str = re.sub(r'\*\*([^*]+)\*\*', r'\1', program_str)
        # Remove list markers at line start: "*   " -> ""
        program_str = re.sub(r'^\*\s+', '', program_str, flags=re.MULTILINE)
        # Remove italic (but be careful with list markers): *text* -> text (if not at line start)
        program_str = re.sub(r'(?<!^)\*([^*\n]+)\*(?!\*)', r'\1', program_str)
        
        # Clean up Query section - extract only the actual query statement
        # Find Query: and everything after it
        query_section_match = re.search(r'Query:\s*(.*?)(?=\n(?:Rules:|Facts:|Predicates:)|$)', program_str, re.IGNORECASE | re.DOTALL)
        if query_section_match:
            query_section = query_section_match.group(1).strip()
            # Look for the actual query pattern: Predicate(Subject, True/False)
            # This pattern matches function calls like Sour(Max, True) or Bright(Stella, False)
            actual_query = re.search(r'(\w+\([A-Za-z]+\w*,\s*(?:True|False)\))', query_section)
            if actual_query:
                query_text = actual_query.group(1)
                # Replace the entire Query section with just the clean query
                program_str = re.sub(r'Query:.*', f'Query:\n{query_text}', program_str, flags=re.IGNORECASE | re.DOTALL)
            else:
                # If no standard pattern found, try to extract the last line that looks like a query
                lines = query_section.split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    # Check if line contains a pattern like "Predicate(Subject, Value)"
                    if re.search(r'\w+\([^)]+\)', line):
                        # Clean it up - remove any markdown or extra text
                        clean_line = re.search(r'(\w+\([^)]+\))', line)
                        if clean_line:
                            program_str = re.sub(r'Query:.*', f'Query:\n{clean_line.group(1)}', program_str, flags=re.IGNORECASE | re.DOTALL)
                            break
        
        return program_str

    def parse_logic_program(self):
        keywords = ['Query:', 'Rules:', 'Facts:', 'Predicates:']
        # Clean the program first
        program_str = self.clean_logic_program(self.logic_program)
        for keyword in keywords:
            try:
                program_str, segment_list = self._parse_segment(program_str, keyword)
                setattr(self, keyword[:-1], segment_list)
            except:
                setattr(self, keyword[:-1], None)

        return self.validate_program()

    def _parse_segment(self, program_str, key_phrase):
        remain_program_str, segment = program_str.split(key_phrase)
        segment_list = segment.strip().split('\n')
        for i in range(len(segment_list)):
            segment_list[i] = segment_list[i].split(':::')[0].strip()
        return remain_program_str, segment_list

    # check if the program is valid; if not, try to fix it
    def validate_program(self):
        if not self.Rules is None and not self.Facts is None:
            if not self.Rules[0] == '' and not self.Facts[0] == '':
                return True
        # try to fix the program
        tmp_rules = []
        tmp_facts = []
        statements = self.Facts if self.Facts is not None else self.Rules
        if statements is None:
            return False
        
        for fact in statements:
            if fact.find('>>>') >= 0: # this is a rule
                tmp_rules.append(fact)
            else:
                tmp_facts.append(fact)
        self.Rules = tmp_rules
        self.Facts = tmp_facts
        return False
    
    def create_fact_file(self, facts):
        with open(os.path.join(self.cache_dir, 'facts.kfb'), 'w') as f:
            for fact in facts:
                # check for invalid facts
                if not fact.find('$x') >= 0:
                    f.write(fact + '\n')

    def create_rule_file(self, rules):
        pyke_rules = []
        for idx, rule in enumerate(rules):
            pyke_rules.append(self.parse_forward_rule(idx + 1, rule))

        with open(os.path.join(self.cache_dir, 'rules.krb'), 'w') as f:
            f.write('\n\n'.join(pyke_rules))

    # example rule: Furry($x, True) && Quite($x, True) >>> White($x, True)
    def parse_forward_rule(self, f_index, rule):
        premise, conclusion = rule.split('>>>')
        premise = premise.strip()
        # split the premise into multiple facts if needed
        premise = premise.split('&&')
        premise_list = [p.strip() for p in premise]

        conclusion = conclusion.strip()
        # split the conclusion into multiple facts if needed
        conclusion = conclusion.split('&&')
        conclusion_list = [c.strip() for c in conclusion]

        # create the Pyke rule
        pyke_rule = f'''fact{f_index}\n\tforeach'''
        for p in premise_list:
            pyke_rule += f'''\n\t\tfacts.{p}'''
        pyke_rule += f'''\n\tassert'''
        for c in conclusion_list:
            pyke_rule += f'''\n\t\tfacts.{c}'''
        return pyke_rule
    
    '''
    for example: Is Marvin from Mars?
    Query: FromMars(Marvin, $label)
    
    Note: Forward chaining rules automatically derive new facts when activated.
    We only need to query facts, not rules, because rules have already been
    applied and their conclusions are stored in facts.
    '''
    def check_specific_predicate(self, subject_name, predicate_name, engine, debug=False):
        results = []
        # Query facts only - forward chaining rules have already been applied
        # and their conclusions are stored in facts
        query_str = f'facts.{predicate_name}({subject_name}, $label)'
        if debug:
            print(f"[DEBUG] Querying: {query_str}")
        
        with engine.prove_goal(query_str) as gen:
            for vars, plan in gen:
                results.append(vars['label'])
                if debug:
                    print(f"[DEBUG] Found result: {vars['label']}")

        # Remove the query to rules - it's incorrect because:
        # 1. Rules are forward chaining rules that derive facts, not queryable predicates
        # 2. When engine.activate('rules') is called, all forward chaining rules run
        #    and their conclusions are automatically added to facts
        # 3. Querying rules.{predicate_name} doesn't make sense in this context

        if debug:
            print(f"[DEBUG] Total results found: {len(results)}")
            if len(results) == 0:
                print(f"[DEBUG] WARNING: No results found for {predicate_name}({subject_name}, $label)")

        if len(results) == 1:
            return results[0]
        elif len(results) > 1:
            # If multiple results found, use the first one
            # (In a well-formed logic program, there should only be one value per predicate per subject)
            if debug:
                print(f"[DEBUG] WARNING: Multiple results found, using first: {results[0]}")
            return results[0]
        elif len(results) == 0:
            return None

    '''
    Input Example: Metallic(Wren, False)
    '''
    def parse_query(self, query):
        pattern = r'(\w+)\(([^,]+),\s*([^)]+)\)'
        match = re.match(pattern, query)
        if match:
            function_name = match.group(1)
            arg1 = match.group(2)
            arg2 = match.group(3)
            arg2 = True if arg2 == 'True' else False
            return function_name, arg1, arg2
        else:
            raise ValueError(f'Invalid query: {query}')

    def execute_program(self, debug=False):
        # delete the compiled_krb dir
        for compiled_dir in self.compiled_dirs:
            if compiled_dir.exists():
                if debug:
                    print(f'removing compiled_krb at {compiled_dir}')
                shutil.rmtree(compiled_dir, ignore_errors=True)
            compiled_dir.mkdir(parents=True, exist_ok=True)
            init_file = compiled_dir / '__init__.py'
            init_file.touch(exist_ok=True)

        # absolute_path = os.path.abspath(complied_krb_dir)
        # print(absolute_path)
        try:
            engine = knowledge_engine.engine(self.cache_dir)
            engine.reset()
            engine.activate('rules')
            engine.get_kb('facts')

            # parse the logic query into pyke query
            query_str = self.Query[0] if isinstance(self.Query, list) else self.Query
            if debug:
                print(f"[DEBUG] Original query: {query_str}")
            
            predicate, subject, value_to_check = self.parse_query(query_str)
            if debug:
                print(f"[DEBUG] Parsed query - Predicate: {predicate}, Subject: {subject}, Value to check: {value_to_check}")
            
            result = self.check_specific_predicate(subject, predicate, engine, debug=debug)
            if debug:
                print(f"[DEBUG] Query result: {result}")
                print(f"[DEBUG] Value to check: {value_to_check}")
                print(f"[DEBUG] Result == Value to check: {result == value_to_check}")
            
            answer = self.answer_map[self.dataset_name](result, value_to_check)
            if debug:
                print(f"[DEBUG] Final answer: {answer}")
        except Exception as e:
            if debug:
                print(f"[DEBUG] Exception occurred: {e}")
                import traceback
                traceback.print_exc()
            return None, e
        
        return answer, ""

    def answer_mapping(self, answer):
        return answer
        
    def answer_map_prontoqa(self, result, value_to_check):
        if result is None:
            return None
        if result == value_to_check:
            return 'A'
        else:
            return 'B'

    def answer_map_proofwriter(self, result, value_to_check):
        if result is None:
            return 'C'
        elif result == value_to_check:
            return 'A'
        else:
            return 'B'


if __name__ == "__main__":

    logic_program = """Predicates:
    Round($x, bool) ::: Is x round?
    Red($x, bool) ::: Is x red?
    Smart($x, bool) ::: Is x smart?
    Furry($x, bool) ::: Is x furry?
    Rough($x, bool) ::: Is x rough?
    Big($x, bool) ::: Is x big?
    White($x, bool) ::: Is x white?
    
    Facts:
    Round(Anne, True) ::: Anne is round.
    Red(Bob, True) ::: Bob is red.
    Smart(Bob, True) ::: Bob is smart.
    Furry(Erin, True) ::: Erin is furry.
    Red(Erin, True) ::: Erin is red.
    Rough(Erin, True) ::: Erin is rough.
    Smart(Erin, True) ::: Erin is smart.
    Big(Fiona, True) ::: Fiona is big.
    Furry(Fiona, True) ::: Fiona is furry.
    Smart(Fiona, True) ::: Fiona is smart.
    
    Rules:
    Smart($x, True) >>> Furry($x, True) ::: All smart things are furry.
    Furry($x, True) >>> Red($x, True) ::: All furry things are red.
    Round($x, True) >>> Rough($x, True) ::: All round things are rough.
    White(Bob, True) >>> Furry(Bob, True) ::: If Bob is white then Bob is furry.
    Red($x, True) && Rough($x, True) >>> Big($x, True) ::: All red, rough things are big.
    Rough($x, True) >>> Smart($x, True) ::: All rough things are smart.
    Furry(Fiona, True) >>> Red(Fiona, True) ::: If Fiona is furry then Fiona is red.
    Round(Bob, True) && Big(Bob, True) >>> Furry(Bob, True) ::: If Bob is round and Bob is big then Bob is furry.
    Red(Fiona, True) && White(Fiona, True) >>> Smart(Fiona, True) ::: If Fiona is red and Fiona is white then Fiona is smart.
    
    Query:
    White(Bob, False) ::: Bob is not white."""

    # Answer: A
    logic_program1 = "Predicates:\nCold($x, bool) ::: Is x cold?\nQuiet($x, bool) ::: Is x quiet?\nRed($x, bool) ::: Is x red?\nSmart($x, bool) ::: Is x smart?\nKind($x, bool) ::: Is x kind?\nRough($x, bool) ::: Is x rough?\nRound($x, bool) ::: Is x round?\n\nFacts:\nCold(Bob, True) ::: Bob is cold.\nQuiet(Bob, True) ::: Bob is quiet.\nRed(Bob, True) ::: Bob is red.\nSmart(Bob, True) ::: Bob is smart.\nKind(Charlie, True) ::: Charlie is kind.\nQuiet(Charlie, True) ::: Charlie is quiet.\nRed(Charlie, True) ::: Charlie is red.\nRough(Charlie, True) ::: Charlie is rough.\nCold(Dave, True) ::: Dave is cold.\nKind(Dave, True) ::: Dave is kind.\nSmart(Dave, True) ::: Dave is smart.\nQuiet(Fiona, True) ::: Fiona is quiet.\n\nRules:\nQuiet($x, True) && Cold($x, True) >>> Smart($x, True) ::: If something is quiet and cold then it is smart.\nRed($x, True) && Cold($x, True) >>> Round($x, True) ::: Red, cold things are round.\nKind($x, True) && Rough($x, True) >>> Red($x, True) ::: If something is kind and rough then it is red.\nQuiet($x, True) >>> Rough($x, True) ::: All quiet things are rough.\nCold($x, True) && Smart($x, True) >>> Red($x, True) ::: Cold, smart things are red.\nRough($x, True) >>> Cold($x, True) ::: If something is rough then it is cold.\nRed($x, True) >>> Rough($x, True) ::: All red things are rough.\nSmart(Dave, True) && Kind(Dave, True) >>> Quiet(Dave, True) ::: If Dave is smart and Dave is kind then Dave is quiet.\n\nQuery:\nKind(Charlie, True) ::: Charlie is kind."

    # Answer: B
    logic_program2 = "Predicates:\nFurry($x, bool) ::: Is x furry?\nNice($x, bool) ::: Is x nice?\nSmart($x, bool) ::: Is x smart?\nYoung($x, bool) ::: Is x young?\nGreen($x, bool) ::: Is x green?\nBig($x, bool) ::: Is x big?\nRound($x, bool) ::: Is x round?\n\nFacts:\nFurry(Anne, True) ::: Anne is furry.\nNice(Anne, True) ::: Anne is nice.\nSmart(Anne, True) ::: Anne is smart.\nYoung(Bob, True) ::: Bob is young.\nNice(Erin, True) ::: Erin is nice.\nSmart(Harry, True) ::: Harry is smart.\nYoung(Harry, True) ::: Harry is young.\n\nRules:\nYoung($x, True) >>> Furry($x, True) ::: Young things are furry.\nNice($x, True) && Furry($x, True) >>> Green($x, True) ::: Nice, furry things are green.\nGreen($x, True) >>> Nice($x, True) ::: All green things are nice.\nNice($x, True) && Green($x, True) >>> Big($x, True) ::: Nice, green things are big.\nGreen($x, True) >>> Smart($x, True) ::: All green things are smart.\nBig($x, True) && Young($x, True) >>> Round($x, True) ::: If something is big and young then it is round.\nGreen($x, True) >>> Big($x, True) ::: All green things are big.\nYoung(Harry, True) >>> Furry(Harry, True) ::: If Harry is young then Harry is furry.\nFurry($x, True) && Smart($x, True) >>> Nice($x, True) ::: Furry, smart things are nice.\n\nQuery:\nGreen(Harry, False) ::: Harry is not green."

    # Answer: C
    logic_program3 = "Predicates:\nChases($x, $y, bool) ::: Does x chase y?\nRough($x, bool) ::: Is x rough?\nYoung($x, bool) ::: Is x young?\nNeeds($x, $y, bool) ::: Does x need y?\nGreen($x, bool) ::: Is x green?\nLikes($x, $y, bool) ::: Does x like y?\nBlue($x, bool) ::: Is x blue?\nRound($x, bool) ::: Is x round?\n\nFacts:\nChases(Cat, Lion, True) ::: The cat chases the lion.\nRough(Cat, True) ::: The cat is rough.\nYoung(Cat, True) ::: The cat is young.\nNeeds(Cat, Lion, True) ::: The cat needs the lion.\nNeeds(Cat, Rabbit, True) ::: The cat needs the rabbit.\nGreen(Dog, True) ::: The dog is green.\nYoung(Dog, True) ::: The dog is young.\nLikes(Dog, Cat, True) ::: The dog likes the cat.\nBlue(Lion, True) ::: The lion is blue.\nGreen(Lion, True) ::: The lion is green.\nChases(Rabbit, Lion, True) ::: The rabbit chases the lion.\nBlue(Rabbit, True) ::: The rabbit is blue.\nRough(Rabbit, True) ::: The rabbit is rough.\nLikes(Rabbit, Dog, True) ::: The rabbit likes the dog.\nNeeds(Rabbit, Dog, True) ::: The rabbit needs the dog.\nNeeds(Rabbit, Lion, True) ::: The rabbit needs the lion.\n\nRules:\nChases($x, Lion, True) >>> Round($x, True) ::: If someone chases the lion then they are round.\nNeeds(Lion, Rabbit, True) && Chases(Rabbit, Dog, True) >>> Likes(Lion, Dog, True) ::: If the lion needs the rabbit and the rabbit chases the dog then the lion likes the dog.\nRound($x, True) && Chases($x, Lion, True) >>> Needs($x, Cat, True) ::: If someone is round and they chase the lion then they need the cat.\nNeeds($x, Cat, True) && Chases($x, Dog, True) >>> Likes($x, Rabbit, True) ::: If someone needs the cat and they chase the dog then they like the rabbit.\nChases($x, Lion, True) && Blue(Lion, True) >>> Round(Lion, True) ::: If someone chases the lion and the lion is blue then the lion is round.\nChases($x, Rabbit, True) >>> Rough($x, True) ::: If someone chases the rabbit then they are rough.\nRough($x, True) && Likes($x, Rabbit, True) >>> Young(Rabbit, True) ::: If someone is rough and they like the rabbit then the rabbit is young.\nChases(Rabbit, Cat, True) && Needs(Cat, Lion, True) >>> Young(Rabbit, True) ::: If the rabbit chases the cat and the cat needs the lion then the rabbit is young.\nRound($x, True) && Needs($x, Cat, True) >>> Chases($x, Dog, True) ::: If someone is round and they need the cat then they chase the dog.\n\nQuery:\nLikes(Lion, Cat, False) ::: The lion does not like the cat."

    # Answer: A
    logic_program4 = "Predicates:\nFurry($x, bool) ::: Is x furry?\nNice($x, bool) ::: Is x nice?\n\nFacts:\nFurry(Anne, True) ::: Anne is furry.\n\nRules:\nFurry($x, True) >>> Nice($x, True) ::: All furry things are nice.\n\nQuery:\nNice(Anne, True) ::: Anne is nice."

    # Answer: B
    logic_program5 = "Predicates:\nFurry($x, bool) ::: Is x furry?\nNice($x, bool) ::: Is x nice?\n\nFacts:\nFurry(Anne, True) ::: Anne is furry.\n\nRules:\nFurry($x, True) >>> Nice($x, True) ::: All furry things are nice.\n\nQuery:\nNice(Anne, False) ::: Anne is not nice."

    # Answer: C
    logic_program6 = "Predicates:\nFurry($x, bool) ::: Is x furry?\nNice($x, bool) ::: Is x nice?\n\nFacts:\nFurry(Anne, True) ::: Anne is furry.\n\nRules:\nFurry($x, False) >>> Nice($x, True) ::: All non-furry things are nice.\n\nQuery:\nNice(Anne, True) ::: Anne is nice."

    # Answer: B
    logic_program7 = """Predicates:
Furry($x, bool) ::: Is x furry?
Nice($x, bool) ::: Is x nice?
Smart($x, bool) ::: Is x smart?
Young($x, bool) ::: Is x young?
Green($x, bool) ::: Is x green?
Big($x, bool) ::: Is x big?
Round($x, bool) ::: Is x round?

Facts:
Furry(Anne, True) ::: Anne is furry.
Nice(Anne, True) ::: Anne is nice.
Smart(Anne, True) ::: Anne is smart.
Young(Bob, True) ::: Bob is young.
Nice(Erin, True) ::: Erin is nice.
Smart(Harry, True) ::: Harry is smart.
Young(Harry, True) ::: Harry is young.

Rules:
Young($x, True) >>> Furry($x, True) ::: Young things are furry.
Nice($x, True) && Furry($x, True) >>> Green($x, True) ::: Nice, furry things are green.
Green($x, True) >>> Nice($x, True) ::: All green things are nice.
Nice($x, True) && Green($x, True) >>> Big($x, True) ::: Nice, green things are big.
Green($x, True) >>> Smart($x, True) ::: All green things are smart.
Big($x, True) && Young($x, True) >>> Round($x, True) ::: If something is big and young then it is round.
Green($x, True) >>> Big($x, True) ::: All green things are big.
Young(Harry, True) >>> Furry(Harry, True) ::: If Harry is young then Harry is furry.
Furry($x, True) && Smart($x, True) >>> Nice($x, True) ::: Furry, smart things are nice.

Query:
Green(Harry, False) ::: Harry is not green."""

    # tests = [logic_program1, logic_program2, logic_program3, logic_program4, logic_program5, logic_program6]

    tests = [logic_program7]
    
    for test in tests:
        pyke_program = Pyke_Program(test, 'ProofWriter')
        print(pyke_program.flag)
        # print(pyke_program.Rules)
        # print(pyke_program.Facts)
        # print(pyke_program.Query)
        # print(pyke_program.Predicates)

        result, error_message = pyke_program.execute_program()
        print(result)

    complied_krb_dir = './compiled_krb'
    if os.path.exists(complied_krb_dir):
        print('removing compiled_krb')
        os.system(f'rm -rf {complied_krb_dir}')