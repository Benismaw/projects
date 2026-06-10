package fr.ensimag.deca.context;

import fr.ensimag.deca.DecacCompiler;
import java.util.HashMap;
import java.util.Map;
import fr.ensimag.deca.tools.SymbolTable.Symbol;
import fr.ensimag.deca.tree.Location;

// A FAIRE: étendre cette classe pour traiter la partie "avec objet" de Déca
/**
 * Environment containing types. Initially contains predefined identifiers, more
 * classes can be added with declareClass().
 *
 * @author gl16
 * @date 01/01/2026
 */
public class EnvironmentType {
    public EnvironmentType(DecacCompiler compiler) {
        
        envTypes = new HashMap<Symbol, TypeDefinition>();
        
        Symbol intSymb = compiler.createSymbol("int");
        INT = new IntType(intSymb);
        envTypes.put(intSymb, new TypeDefinition(INT, Location.BUILTIN));

        Symbol floatSymb = compiler.createSymbol("float");
        FLOAT = new FloatType(floatSymb);
        envTypes.put(floatSymb, new TypeDefinition(FLOAT, Location.BUILTIN));

        Symbol voidSymb = compiler.createSymbol("void");
        VOID = new VoidType(voidSymb);
        envTypes.put(voidSymb, new TypeDefinition(VOID, Location.BUILTIN));

        Symbol booleanSymb = compiler.createSymbol("boolean");
        BOOLEAN = new BooleanType(booleanSymb);
        envTypes.put(booleanSymb, new TypeDefinition(BOOLEAN, Location.BUILTIN));

        Symbol stringSymb = compiler.createSymbol("string");
        STRING = new StringType(stringSymb);

        // Ajout du type NULL
        this.NULL = new NullType(compiler.symbolTable.create("null"));

        //inclure les declarations de Object
        Symbol objectSymb = compiler.createSymbol("Object");

        ClassType objectType = new ClassType(objectSymb, Location.BUILTIN, null);

        ClassDefinition objectDef = objectType.getDefinition();

        Signature equalsSig = new Signature();
        equalsSig.add(objectType);

        MethodDefinition equalsDef = new MethodDefinition(
                this.BOOLEAN,
                Location.BUILTIN,
                equalsSig,
                1
        );

        try {
            objectDef.getMembers().declare(compiler.createSymbol("equals"), equalsDef);
            objectDef.setNumberOfMethods(1); 
        } catch (EnvironmentExp.DoubleDefException e) {
        }

        envTypes.put(objectSymb, objectDef);

    }

    private final Map<Symbol, TypeDefinition> envTypes;

    public TypeDefinition defOfType(Symbol s) {
        return envTypes.get(s);
    }


    public void declare(Symbol name,TypeDefinition def) throws ContextualError{
        if (envTypes.containsKey(name)) {
        throw new ContextualError(
            "Classe " + name.getName() + " déjà définie",
            def.getLocation()
        );
    }
        this.envTypes.put(name,def);
    }

    //ajout de methode pour la passe 2
    public boolean subtype(Type t1, Type t2) {
         // Implémentation des règles de sous-typage 
         if (t1.sameType(t2)){
            return true;
         }
         if (t1.isNull()){
             return t2.isClass() || t2.isNull();
         }
          if (t1.isClass() && t2.isClass()) {
                ClassType ct1 = (ClassType) t1;
                ClassType ct2 = (ClassType) t2;
                return ct1.isSubClassOf(ct2);
            }
         return false ;
    }
    public TypeDefinition get(Symbol s) {
       return envTypes.get(s);
    }
    public final VoidType    VOID;
    public final IntType     INT;
    public final FloatType   FLOAT;
    public final StringType  STRING;
    public final BooleanType BOOLEAN;
    public final Type NULL;
}
