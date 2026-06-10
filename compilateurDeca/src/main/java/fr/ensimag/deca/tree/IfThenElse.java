package fr.ensimag.deca.tree;

import java.io.PrintStream;

import org.apache.commons.lang.Validate;

import fr.ensimag.ARM.ARMImmediate;
import fr.ensimag.ARM.ARMLabel;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMB;
import fr.ensimag.ARM.Instructions.ARMBeq;
import fr.ensimag.ARM.Instructions.ARMCmp;
import fr.ensimag.ARM.Instructions.ARMMov;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.ImmediateInteger;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.instructions.BEQ;
import fr.ensimag.ima.pseudocode.instructions.BRA;
import fr.ensimag.ima.pseudocode.instructions.CMP;

/**
 * Full if/else if/else statement.
 *
 * @author gl16
 * @date 01/01/2026
 */
public class IfThenElse extends AbstractInst {
    
    private final AbstractExpr condition; 
    private final ListInst thenBranch;
    private ListInst elseBranch;

    public IfThenElse(AbstractExpr condition, ListInst thenBranch, ListInst elseBranch) {
        Validate.notNull(condition);
        Validate.notNull(thenBranch);
        Validate.notNull(elseBranch);
        this.condition = condition;
        this.thenBranch = thenBranch;
        this.elseBranch = elseBranch;
    }
    
    @Override
    protected void verifyInst(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass, Type returnType)
            throws ContextualError {
        condition.verifyCondition(compiler, localEnv, currentClass);
        thenBranch.verifyListInst(compiler, localEnv, currentClass, returnType);
        elseBranch.verifyListInst(compiler, localEnv, currentClass, returnType);
    }

    @Override
    protected void codeGenInst(DecacCompiler compiler) {

        Label elseLabel = compiler.getLabelManager().uniqueLabel("else");
        Label endLabel = compiler.getLabelManager().uniqueLabel("end_if");

        condition.codeGenExpr(compiler, Register.getR(2));

        // Condition
        compiler.addInstruction(new CMP(new ImmediateInteger(0), Register.getR(2)));
        compiler.addInstruction(new BEQ(elseLabel));

        // Code du if
        thenBranch.codeGenListInst(compiler);
        compiler.addInstruction(new BRA(endLabel));

        // Sinon
        compiler.addLabel(elseLabel);
        elseBranch.codeGenListInst(compiler);

        // Fin
        compiler.addLabel(endLabel);

    }

    @Override
    public void decompile(IndentPrintStream s) {
        s.print("if (");
        condition.decompile(s);
        s.print(") {");
        s.println();
        s.indent();
        thenBranch.decompile(s);
        s.unindent();
        s.print("}");
        s.print(" else {");
        s.println();
        s.indent();
        elseBranch.decompile(s);
        s.unindent();
        s.print("}");
    }

    @Override
    protected
    void iterChildren(TreeFunction f) {
        condition.iter(f);
        thenBranch.iter(f);
        elseBranch.iter(f);
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        condition.prettyPrint(s, prefix, false);
        thenBranch.prettyPrint(s, prefix, false);
        elseBranch.prettyPrint(s, prefix, true);
    }

    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
        var prog = compiler.getARMProgram();

        String LabelElse = "else_" + this.hashCode();
        String LabelEnd = "endif_" + this.hashCode();

        condition.codeGenExprARM(compiler, ARMRegister.R0);

        prog.addInstruction(new ARMMov(ARMRegister.R1, new ARMImmediate(0)));
        prog.addInstruction(new ARMCmp(ARMRegister.R0, ARMRegister.R1));

        prog.addInstruction(new ARMBeq(LabelElse));

        thenBranch.codeGenListInstARM(compiler);

        prog.addInstruction(new ARMB(LabelEnd));

        prog.addInstruction(new ARMLabel(LabelElse));
        elseBranch.codeGenListInstARM(compiler);

        prog.addInstruction(new ARMLabel(LabelEnd));
    }
}
