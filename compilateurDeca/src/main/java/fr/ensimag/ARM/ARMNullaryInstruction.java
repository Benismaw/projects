package fr.ensimag.ARM;

/**
 * Instruction without operand.
 *
 * @author Ensimag
 * @date 01/01/2026
 */
public abstract class ARMNullaryInstruction extends ARMInstruction {
    
    @Override
    public String toString() {
       return " "; // no operand
    }
}
