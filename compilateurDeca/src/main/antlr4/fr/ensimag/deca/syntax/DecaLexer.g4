lexer grammar DecaLexer;

options {
   language=Java;
   // Tell ANTLR to make the generated lexer class extend the
   // the named class, which is where any supporting code and
   // variables will be placed.
   superClass = AbstractDecaLexer;
}

@members {
 private void lexError(String msg) {
         String source = getSourceName();
         int line = _tokenStartLine;
         String fullMsg = source + ":" + line + ": " + msg;
         throw new RuntimeException(fullMsg);
    }
}

// Deca lexer rules.

/* ---------- Mots réservés ---------- */
ASM        : 'asm';
CLASS      : 'class';
EXTENDS    : 'extends';
ELSE       : 'else';
FALSE      : 'false';
IF         : 'if';
INSTANCEOF : 'instanceof';
NEW        : 'new';
NULL       : 'null';
READINT    : 'readInt';
READFLOAT  : 'readFloat';
PRINT      : 'print';
PRINTLN    : 'println';
PRINTLNX   : 'printlnx';
PRINTX     : 'printx';
PROTECTED  : 'protected';
RETURN     : 'return';
THIS       : 'this';
TRUE       : 'true';
WHILE      : 'while';


/* ---------- Identificateurs ---------- */
fragment LETTER : [a-zA-Z];
fragment DIGIT  : [0-9];
IDENT           : (LETTER | '$' | '_') (LETTER | DIGIT | '$' | '_')*;


/* ---------- Symboles spéciaux ---------- */
// 2-caractères
EQEQ : '==';
NEQ  : '!=';
GEQ  : '>=';
LEQ  : '<=';
AND  : '&&';
OR   : '||';

// 1-caractère
LT      : '<';
GT      : '>';
EQUALS  : '=';
PLUS    : '+';
MINUS   : '-';
TIMES   : '*';
SLASH   : '/';
PERCENT : '%';
DOT     : '.';
COMMA   : ',';
OPARENT : '(';
CPARENT : ')';
OBRACE  : '{';
CBRACE  : '}';
EXCLAM  : '!';
SEMI    : ';';


/* ---------- Littéraux entiers ---------- */
fragment POSITIVE_DIGIT : [1-9];
INT                     : '0' | POSITIVE_DIGIT DIGIT*;


/* ---------- Littéraux flottants ---------- */
fragment NUM      : DIGIT+;
fragment SIGN     : [+-]?;
fragment EXP      : [Ee] SIGN NUM;
fragment DEC      : NUM '.' NUM;
fragment FLOATDEC :  DEC EXP? [fF]?;
fragment DIGITHEX :  [0-9a-fA-F];
fragment NUMHEX   : DIGITHEX+;
fragment FLOATHEX : ('0x'|'0X') NUMHEX '.' NUMHEX [Pp] SIGN NUM [fF]?;
FLOAT             : FLOATDEC | FLOATHEX;


/* ---------- Chaînes de caractères ---------- */
fragment EOL        : '\r\n' | '\n' | '\r';
fragment STRING_CAR : ~["\\\r\n];
STRING              : '"' (STRING_CAR | '\\"' | '\\\\')* '"';
MULTI_LINE_STRING   : '"' (STRING_CAR | EOL | '\\"' | '\\\\')* '"';
UNTERMINATED_STRING : '"' (STRING_CAR | EOL | '\\"' | '\\\\')* { lexError("Chaine de caracteres non terminee"); }
    ;

/* ---------- Commentaires et séparateurs ---------- */
BLOCK_COMMENT : '/*' .*? '*/' -> skip;
LINE_COMMENT  : '//' ~[\r\n]* -> skip;
SEP           : [ \t\r\n]+ -> skip;


/* ---------- Inclusion de fichier ---------- */
fragment FILENAME : (LETTER | DIGIT | '.' | '-' | '_')+;
INCLUDE           : '#include' (' ')* '"' FILENAME '"' {doInclude(getText());} -> skip;


/* ---------- Erreur ---------- */
ERROR_CHAR : . { lexError("Caractère invalide: " + getText()); };
