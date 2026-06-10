package fr.ensimag.deca.tree;

import java.io.PrintStream;

import org.apache.commons.lang.Validate;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.Instructions.ARMLdrFloat;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.ImmediateFloat;
import fr.ensimag.ima.pseudocode.instructions.LOAD;

/**
 * Single precision, floating-point literal
 *
 * @author gl16
 * @date 01/01/2026
 */
public class FloatLiteral extends AbstractExpr {

    public float getValue() {
        return value;
    }

    private float value;

    public FloatLiteral(float value) {
        Validate.isTrue(!Float.isInfinite(value),
                "literal values cannot be infinite");
        Validate.isTrue(!Float.isNaN(value),
                "literal values cannot be NaN");
        this.value = value;
    }
    
     @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register){
        float val=getValue();
        compiler.addInstruction(new LOAD(new ImmediateFloat(val),register));
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass) throws ContextualError {
        this.setType(compiler.environmentType.FLOAT);
        return compiler.environmentType.FLOAT;
    }


    @Override
    public void decompile(IndentPrintStream s) {
        s.print(java.lang.Float.toHexString(value));
    }

    @Override
    String prettyPrintNode() {
        return "Float (" + getValue() + ")";
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        // leaf node => nothing to do
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        // leaf node => nothing to do
    }

    private static int counter;

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        String label = "flt_" + (counter++);
        compiler.getARMProgram().addData(
            label + ": .float " + value 
        );
        compiler.getARMProgram().addInstruction(
            new ARMLdrFloat(register, label)
        );
    }

    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
        
    }

}
