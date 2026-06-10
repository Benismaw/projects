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
 *
 * @author gl16
 * @date 01/01/2026
 */
public class While extends AbstractInst {
    private AbstractExpr condition;
    private ListInst body;

    public AbstractExpr getCondition() {
        return condition;
    }

    public ListInst getBody() {
        return body;
    }

    public While(AbstractExpr condition, ListInst body) {
        Validate.notNull(condition);
        Validate.notNull(body);
        this.condition = condition;
        this.body = body;
    }

    @Override
    protected void codeGenInst(DecacCompiler compiler) {
        Label debutWhile = compiler.getLabelManager().uniqueLabel("while_debut");
        Label finWhile   = compiler.getLabelManager().uniqueLabel("while_fin");

        compiler.addLabel(debutWhile);

        condition.codeGenExpr(compiler, Register.getR(2));

        compiler.addInstruction(new CMP(new ImmediateInteger(0), Register.getR(2)));
        compiler.addInstruction(new BEQ(finWhile));

        body.codeGenListInst(compiler);

        compiler.addInstruction(new BRA(debutWhile));

        compiler.addLabel(finWhile);
    }

    @Override
    protected void verifyInst(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass, Type returnType)
            throws ContextualError {
        this.condition.verifyCondition(compiler, localEnv, currentClass);
        this.body.verifyListInst(compiler, localEnv, currentClass, returnType);
    }

    @Override
    public void decompile(IndentPrintStream s) {
        s.print("while (");
        getCondition().decompile(s);
        s.println(") {");
        s.indent();
        getBody().decompile(s);
        s.unindent();
        s.print("}");
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        condition.iter(f);
        body.iter(f);
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        condition.prettyPrint(s, prefix, false);
        body.prettyPrint(s, prefix, true);
    }

    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
        var prog = compiler.getARMProgram();

        String labelDebut = "while_debut_" + this.hashCode();
        String labelFin = "while_fin_" + this.hashCode();

        prog.addInstruction(new ARMLabel(labelDebut));

        condition.codeGenExprARM(compiler, ARMRegister.R0);

        prog.addInstruction(new ARMMov(ARMRegister.R1, new ARMImmediate(0)));
        prog.addInstruction(new ARMCmp(ARMRegister.R0, ARMRegister.R1));

        prog.addInstruction(new ARMBeq(labelFin));

        body.codeGenListInstARM(compiler);

        prog.addInstruction(new ARMB(labelDebut));

        prog.addInstruction(new ARMLabel(labelFin));
    }


}
