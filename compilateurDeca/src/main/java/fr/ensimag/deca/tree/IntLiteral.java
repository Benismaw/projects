package fr.ensimag.deca.tree;

import java.io.PrintStream;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMImmediate;
import fr.ensimag.ARM.Instructions.ARMMov;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.ImmediateInteger;
import fr.ensimag.ima.pseudocode.instructions.LOAD;

/**
 * Integer literal
 *
 * @author gl16
 * @date 01/01/2026
 */
public class IntLiteral extends AbstractExpr {
    public int getValue() {
        return value;
    }

    private int value;

    public IntLiteral(int value) {
        this.value = value;
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass) throws ContextualError {
        // Decoration de l'AST
        this.setType(compiler.environmentType.INT);
        return compiler.environmentType.INT;
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        compiler.addInstruction(new LOAD(new ImmediateInteger(value), register));
    }


    @Override
    String prettyPrintNode() {
        return "Int (" + getValue() + ")";
    }

    @Override
    public void decompile(IndentPrintStream s) {
        s.print(Integer.toString(value));
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        // leaf node => nothing to do
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        // leaf node => nothing to do
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        compiler.getARMProgram().addInstruction(
            new ARMMov(register, new ARMImmediate(value))
        );
    }

    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
        
    }

}
