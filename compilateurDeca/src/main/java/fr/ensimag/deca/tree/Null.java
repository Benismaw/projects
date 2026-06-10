package fr.ensimag.deca.tree;

import java.io.PrintStream;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.NullOperand;
import fr.ensimag.ima.pseudocode.instructions.LOAD;

public class Null extends AbstractExpr {

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass) throws ContextualError {
        Type type = compiler.environmentType.NULL; 
        this.setType(type);
        return type;
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        compiler.addInstruction(new LOAD(new NullOperand(), register));
    }

    @Override
    public void decompile(IndentPrintStream s) {
        s.print("null"); 
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        // Le nœud Null est une feuille, il n'a pas d'enfants 
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        // Pas d'enfants à afficher
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        //todo
    }
}