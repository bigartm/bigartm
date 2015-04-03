import re

def CheckSpaceBetweenLangleAndSemicolon(filename, clean_lines, linenum, error):
  """Checks for <:: declarations.
  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]

  # Matches <:: which is illegal in some compilers.
  matched = re.findall('<::', line)
  if matched:
    error(filename, linenum, 'whitespace/declaration', 3,
          'Declaration has no space in <:: (between l-angle and semicolon)')
