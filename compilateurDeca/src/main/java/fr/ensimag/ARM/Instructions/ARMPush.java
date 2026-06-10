package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMPush extends ARMInstruction {

    private final ARMRegister reg;

    public ARMPush(ARMRegister reg) {
        this.reg = reg;
    }

    @Override
    public String toString() {
        return "push {" + reg + "}";
    }    
}
