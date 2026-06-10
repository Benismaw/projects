package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMInstruction;

public class ARMLdrFloat extends ARMInstruction{
    private final ARMGPRegister dst;
    private final String label;

    public ARMLdrFloat(ARMGPRegister dst, String label) {
        this.dst = dst;
        this.label = label;
    }

    @Override 
    public String toString() {
        return "ldr " + dst + ", =" + label + "\n" + "\tvldr.f32 s0, [r0]";   
    }
}
