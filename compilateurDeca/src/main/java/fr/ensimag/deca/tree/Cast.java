package fr.ensimag.deca.tree;


import java.io.PrintStream;

import org.apache.commons.lang.Validate;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.GPRegister;
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
import fr.ensimag.ima.pseudocode.instructions.POP;
import fr.ensimag.ima.pseudocode.instructions.PUSH;

/**
 *
 * @author gl16
 * @date 07/01/2026
 */
public class Cast extends AbstractUnaryExpr{

    private AbstractIdentifier type;

    public Cast(AbstractIdentifier type, AbstractExpr operande){
        super(operande);
        Validate.notNull(type);
        this.type=type;
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv, ClassDefinition currentClass) throws ContextualError {

        Type typeToCast = type.verifyType(compiler);
        Type typeExpr = getOperand().verifyExpr(compiler, localEnv, currentClass);

        if (!Type.castCompatible(compiler.environmentType, typeToCast, typeExpr)) {
            throw new ContextualError("Cast invalide de " + typeExpr + " vers " + typeToCast + " (règle 3.39)", getLocation());
        }
        setType(typeToCast);
        return typeToCast;
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        getOperand().codeGenExpr(compiler, register);
        ClassDefinition classDef = type.getClassDefinition(); // target class is saved as a type in the environment

        //labels
        Label loopLabel = compiler.getLabelManager().uniqueLabel("cast_loop");
        Label successLabel = compiler.getLabelManager().uniqueLabel("cast_success");
        Label endLabel = compiler.getLabelManager().uniqueLabel("cast_end");

        Label castErrorLabel = new Label("cast_error");

        //register for target class
        GPRegister targetClassReg = Register.R0;
        if (register.equals(Register.R0)) {
            targetClassReg = Register.R1;
        }

        // Check if null
        compiler.addInstruction(new CMP(new NullOperand(), register));
        compiler.addInstruction(new BEQ(endLabel));

        // save original object register in stack
        compiler.addInstruction(new PUSH(register));
        
        // objects dynamic type and target class loading
        compiler.addInstruction(new LOAD(new RegisterOffset(0, register), register)); // loads the Vtable address stored in the object, which is a reference to its class
        compiler.addInstruction(new LEA(classDef.getVtableAddr(), targetClassReg)); // loads the address of the target class Vtable


        //loop: compares object (and its hierarchy) to target class
        compiler.addLabel(loopLabel);

        compiler.addInstruction(new CMP(targetClassReg, register));
        compiler.addInstruction(new BEQ(successLabel));

            //load parent
            compiler.addInstruction(new LOAD(new RegisterOffset(0, register), register)); // loads the parent class Vtable address into the object register
            compiler.addInstruction(new BNE(loopLabel));

        //case of failure
        compiler.addInstruction(new BRA(castErrorLabel));

        //case of success
        compiler.addLabel(successLabel);
        compiler.addInstruction(new POP(register));

        //end
        compiler.addLabel(endLabel);
   }


    @Override
    public void decompile(IndentPrintStream s) {
        s.print("(");
        type.decompile(s);
        s.print(") (");
        getOperand().decompile(s);
        s.print(")");
    }

    @Override
    protected String getOperatorName() {
        return "cast";
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        type.prettyPrint(s, prefix, false);
        getOperand().prettyPrint(s, prefix, true);
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        type.iter(f);
        getOperand().iter(f);
    }

    
    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        Type targetType = getType();
        Type exprType = getOperand().getType();

        // int vers float
        if (exprType.isInt() && targetType.isFloat()) {
            getOperand().codeGenExprARM(compiler, register);
            compiler.getARMProgram().addRaw("""
                vmov s0, r0
                vcvt.f32.s32 s0, s0
            """);
            return;
        }
        
        // tous les autres casts sont no-op
        getOperand().codeGenExprARM(compiler, register);
    }

    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
}
}