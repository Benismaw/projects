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
import fr.ensimag.ima.pseudocode.ImmediateInteger;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.NullOperand;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.RegisterOffset;
import fr.ensimag.ima.pseudocode.instructions.BEQ;
import fr.ensimag.ima.pseudocode.instructions.BNE;
import fr.ensimag.ima.pseudocode.instructions.BRA;
import fr.ensimag.ima.pseudocode.instructions.CMP;
import fr.ensimag.ima.pseudocode.instructions.LEA;
import fr.ensimag.ima.pseudocode.instructions.LOAD;

/**
 * Opérateur instanceof
 */
public class Instanceof extends AbstractExpr {
    private AbstractExpr expr;
    private AbstractIdentifier type;

    public Instanceof(AbstractExpr expr, AbstractIdentifier type) {
        this.expr = expr;
        this.type = type;
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
                           ClassDefinition currentClass) throws ContextualError {

        // Vérifier l'expression gauche
        Type typeExpr = expr.verifyExpr(compiler, localEnv, currentClass);

        // Vérifier le type droit
        Type typeCheck = type.verifyType(compiler);

        // Le membre gauche doit être une classe ou null
        if (!typeExpr.isClassOrNull()) {
            throw new ContextualError("Le membre gauche de instanceof doit être un objet ou null. Type trouvé : " + typeExpr, getLocation());
        }

        // Le membre droit doit être un type classe
        if (!typeCheck.isClass()) {
            throw new ContextualError("Le membre droit de instanceof doit être une classe. Type trouvé : " + typeCheck, getLocation());
        }

        setType(compiler.environmentType.BOOLEAN);
        return compiler.environmentType.BOOLEAN;
    }

    @Override
    public void decompile(IndentPrintStream s) {
        s.print("(");
        expr.decompile(s);
        s.print(" instanceof ");
        type.decompile(s);
        s.print(")");
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        expr.iter(f);
        type.iter(f);
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        expr.prettyPrint(s, prefix, false);
        type.prettyPrint(s, prefix, true);
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        expr.codeGenExpr(compiler, register); //
        ClassDefinition classDef = type.getClassDefinition(); // target class is saved as a type in the environment

        //labels
        Label loopLabel = compiler.getLabelManager().uniqueLabel("InstanceOf_loop");
        Label successLabel = compiler.getLabelManager().uniqueLabel("InstanceOf_success");
        Label endLabel = compiler.getLabelManager().uniqueLabel("InstanceOf_end");
        Label failureLabel = compiler.getLabelManager().uniqueLabel("InstanceOf_failure");

        //register for target class
        GPRegister targetClassReg = Register.R0;
        if (register.equals(Register.R0)) {
            targetClassReg = Register.R1;
        }

        // Check if null
        compiler.addInstruction(new CMP(new NullOperand(), register));
        compiler.addInstruction(new BEQ(failureLabel));

        // objects dynamic type
        compiler.addInstruction(new LOAD(new RegisterOffset(0, register), register)); // loads the Vtable address stored in the object, which is a reference to its class
        
        compiler.addInstruction(new LEA(classDef.getVtableAddr(), targetClassReg)); // loads the address of the target class Vtable

        //loop: compares object (and its hierarchy) to target class
        compiler.addLabel(loopLabel);

        compiler.addInstruction(new CMP(targetClassReg, register));
        compiler.addInstruction(new BEQ(successLabel));

        //load parent
        compiler.addInstruction(new LOAD(new RegisterOffset(0, register), register)); // loads the parent class Vtable address into the object register
        compiler.addInstruction(new CMP(new NullOperand(), register));
        compiler.addInstruction(new BNE(loopLabel));

        //case of failure
        compiler.addLabel(failureLabel);
        compiler.addInstruction(new LOAD(new ImmediateInteger(0), register));
        compiler.addInstruction(new BRA(endLabel));

        //case of success
        compiler.addLabel(successLabel);
        compiler.addInstruction(new LOAD(new ImmediateInteger(1), register));

        //end
        compiler.addLabel(endLabel);
   }

    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'codeGenInstARM'");
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'codeGenExprARM'");
    }
}