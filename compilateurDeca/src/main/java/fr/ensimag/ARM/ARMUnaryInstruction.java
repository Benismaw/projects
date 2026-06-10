package fr.ensimag.ARM;

import org.apache.commons.lang.Validate;

/**
 * Instruction with a single operand.
 *
 * @author Ensimag
 * @date 01/01/2026
 */
public abstract class ARMUnaryInstruction extends ARMInstruction {
    private ARMOperand operand;

    protected ARMUnaryInstruction(ARMOperand operand) {
        Validate.notNull(operand);
        this.operand = operand;
    }

    public ARMOperand getOperand() {
        return operand;
    }

    @Override
    public String toString() {
        return " " + operand.toString();
    }

}
