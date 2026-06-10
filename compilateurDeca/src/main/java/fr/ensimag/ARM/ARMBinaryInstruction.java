package fr.ensimag.ARM;

import org.apache.commons.lang.Validate;

/**
 * Base class for instructions with 2 operands.
 *
 * @author Ensimag
 * @date 01/01/2026
 */
public class ARMBinaryInstruction extends ARMInstruction {
    private ARMOperand operand1, operand2;

    public ARMOperand getOperand1() {
        return operand1;
    }

    public ARMOperand getOperand2() {
        return operand2;
    }

    protected ARMBinaryInstruction(ARMOperand op1, ARMOperand op2) {
        Validate.notNull(op1);
        Validate.notNull(op2);
        this.operand1 = op1;
        this.operand2 = op2;
    }

    @Override
    public String toString() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'toString'");
    }
}
