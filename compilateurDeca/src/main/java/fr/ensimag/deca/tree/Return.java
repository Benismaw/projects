package fr.ensimag.deca.tree;

import java.io.PrintStream;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.instructions.BRA;

public class Return extends AbstractInst {
    private AbstractExpr expression;

    public Return(AbstractExpr expression){
        this.expression=expression;
    }
    @Override
    protected void verifyInst(DecacCompiler compiler,
            EnvironmentExp localEnv, ClassDefinition currentClass, Type returnType) throws ContextualError {
        
        Type typeExpr = this.expression.verifyExpr(compiler, localEnv, currentClass);

        if (returnType.isVoid()) {
            throw new ContextualError("Impossible de retourner une valeur dans une méthode de type void (règle 3.24)", getLocation());
        }

        this.expression = this.expression.verifyRValue(compiler, localEnv, currentClass, returnType);
    }

    @Override
    protected void codeGenInst(DecacCompiler compiler){
        expression.codeGenExpr(compiler, Register.R0);
        try {
            Label endLabel = compiler.getCurrentMethodEndLabel();
            compiler.addInstruction(new BRA(endLabel));
        } catch (Exception ignored) {

        }
    }

    @Override
    public void decompile(IndentPrintStream s) {
        s.print("return ");
        expression.decompile(s);
        s.print(";");
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        expression.iter(f);
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        expression.prettyPrint(s, prefix, true);
    }
    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'codeGenInstARM'");
    }
}