package fr.ensimag.ARM;

/**
 * Abstract representation of an ARM instruction
 * Each instruction knows how to display itself as valid ARM assembly
 * 
 */

public abstract class ARMInstruction {
    String getName() {
        return this.getClass().getSimpleName();
    }
    // abstract void displayOperands(PrintStream s);
    // void display(PrintStream s) {
    //     s.print(getName());
    //     displayOperands(s);
    // }

    @Override
    public abstract String toString();
}
