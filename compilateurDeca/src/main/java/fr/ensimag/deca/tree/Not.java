package fr.ensimag.deca.tree;

import fr.ensimag.ARM.ARMGPRegister;
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
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.ImmediateInteger;
import fr.ensimag.ima.pseudocode.instructions.CMP;
import fr.ensimag.ima.pseudocode.instructions.SEQ;

/**
 *
 * @author gl16
 * @date 01/01/2026
 */
public class Not extends AbstractUnaryExpr {

    public Not(AbstractExpr operand) {
        super(operand);
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass) throws ContextualError {
        // On vérifie le type
        Type type = getOperand().verifyExpr(compiler, localEnv, currentClass);

        // On vérifie que c'est un booléen
        if (!type.isBoolean()) {
            throw new ContextualError("L'opérande du not doit être de type booléen. Type trouvé : " + type + " (règle 3.37)", getLocation());
        }

        setType(type);
        return type;
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        getOperand().codeGenExpr(compiler, register);
        //comparer la valeur avec 0 
        compiler.addInstruction(new CMP(new ImmediateInteger(0),register));
        //mettre à 1 si egalité (si c'est faux), sinon à 0
        compiler.addInstruction(new SEQ(register));
    }


    @Override
    protected String getOperatorName() {
        return "!";
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        var prog = compiler.getARMProgram();

        String labelTrue = "not_true_" + this.hashCode();
        String labelEnd = "not_end_" + this.hashCode();

        getOperand().codeGenExprARM(compiler, register);

        prog.addInstruction(new ARMMov(ARMRegister.R1, new ARMImmediate(0)));
        prog.addInstruction(new ARMCmp(register, ARMRegister.R1));

        prog.addInstruction(new ARMBeq(labelTrue));

        // False case
        prog.addInstruction(new ARMMov(register, new ARMImmediate(0)));
        prog.addInstruction(new ARMB(labelEnd));

        prog.addInstruction(new ARMLabel(labelTrue));
        prog.addInstruction(new ARMMov( register, new ARMImmediate(1)));
        prog.addInstruction(new ARMLabel(labelEnd));

    }
}
