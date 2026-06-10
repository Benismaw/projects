package fr.ensimag.deca.tree;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.deca.tools.SymbolTable;
import fr.ensimag.ima.pseudocode.*;
import fr.ensimag.ima.pseudocode.instructions.*;
import org.apache.log4j.Logger;

/**
 *
 * @author gl16
 * @date 01/01/2026
 */
public class ListDeclClass extends TreeList<AbstractDeclClass> {
    private static final Logger LOG = Logger.getLogger(ListDeclClass.class);
    
    @Override
    public void decompile(IndentPrintStream s) {
        for (AbstractDeclClass c : getList()) {
            c.decompile(s);
            s.println();
        }
    }

    /**
     * Pass 1 of [SyntaxeContextuelle]
     */
    void verifyListClass(DecacCompiler compiler) throws ContextualError {
        //LOG.debug("verify listClass: start");

        // Passe 1 : déclarer les classes
        for (AbstractDeclClass c : getList()) {
                // juste créer la définition vide pour réserver le nom
                c.verifyClass(compiler); // uniquement la déclaration, pas encore les membres
            }
        }

    /**
     * Pass 2 of [SyntaxeContextuelle]
     */
    public void verifyListClassMembers(DecacCompiler compiler) throws ContextualError {
        for (AbstractDeclClass c : getList()) {
            c.verifyClassMembers(compiler);
        }
    }
    
    /**
     * Pass 3 of [SyntaxeContextuelle]
     */
    public void verifyListClassBody(DecacCompiler compiler) throws ContextualError {
        for (AbstractDeclClass c : getList()) {
            c.verifyClassBody(compiler);
        }
    }

    // Passe 1
    public void codeGenListDeclClassVTable(DecacCompiler compiler) {

    // Initialisation de la classe Object (La Racine)
    SymbolTable.Symbol objectSymbol = compiler.symbolTable.create("Object");
    ClassDefinition objectDef = (ClassDefinition) compiler.environmentType.defOfType(objectSymbol);

    // Allocation dans la Pile Globale (GB) - Zone Statique
    // On alloue la première case
    DAddr objectAddr = compiler.getStackManager().allocGlobalVar();
    objectDef.setVtableAddr(objectAddr);
    
    // On alloue la deuxième case
    DAddr equalsAddr = compiler.getStackManager().allocGlobalVar();

    compiler.addComment("Code de la table des méthodes de Object");

    // Remplissage de la VTable de Object

    // Slot 0 : Super classe (null)
    compiler.addInstruction(new LOAD(new NullOperand(), Register.R0));
    compiler.addInstruction(new STORE(Register.R0, objectAddr));

    // On utilise LEA pour charger l'adresse du label
    Label labelEquals = new Label("code.Object.equals");
    compiler.addInstruction(new LOAD(new LabelOperand(labelEquals), Register.R0));
    // On stocke directement dans l'adresse allouée pour le slot 1
    compiler.addInstruction(new STORE(Register.R0, equalsAddr));

    // Génération pour les classes définies par l'utilisateur
    for (AbstractDeclClass c : getList()) {
        c.codeGenVTable(compiler);
    }
}

    // Passe 2
    public void codeGenListDeclClassBody(DecacCompiler compiler) {
        // Generation de l'initialisation de Object
        compiler.addLabel(new Label("init.Object"));
        compiler.addInstruction(new RTS());

        // Generation de la methode equals de Object
        compiler.addLabel(new Label("code.Object.equals"));
        compiler.addInstruction(new RTS());

        for (AbstractDeclClass c : getList()) {
            ((DeclClass)c).codeGenInit(compiler);
            c.codeGenMethod(compiler);
        }
    }


}
