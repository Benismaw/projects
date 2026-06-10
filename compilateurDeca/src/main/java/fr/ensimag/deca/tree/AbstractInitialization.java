package fr.ensimag.deca.tree;

import fr.ensimag.ARM.ARMDAddr;
import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.ima.pseudocode.DAddr;
import fr.ensimag.ima.pseudocode.GPRegister;

/**
 * Initialization (of variable, field, ...)
 *
 * @author gl16
 * @date 01/01/2026
 */
public abstract class AbstractInitialization extends Tree {
    
    /**
     * Implements non-terminal "initialization" of [SyntaxeContextuelle] in pass 3
     * @param compiler contains "env_types" attribute
     * @param t corresponds to the "type" attribute
     * @param localEnv corresponds to the "env_exp" attribute
     * @param currentClass 
     *          corresponds to the "class" attribute (null in the main bloc).
     */
    protected abstract void verifyInitialization(DecacCompiler compiler,
            Type t, EnvironmentExp localEnv, ClassDefinition currentClass)
            throws ContextualError;
    protected abstract void codeGenInitialization(DecacCompiler compiler, GPRegister register, DAddr addr);

    /**
     * Génère le code pour l'initialisation d'un champ d'objet.
     * @param compiler
     * @param index Index du champ dans l'objet (offset dans le tas)
     */
    protected abstract void codeGenField(DecacCompiler compiler, int index);
    

    protected abstract void codeGenInitializationARM(DecacCompiler compiler, ARMGPRegister register, ARMDAddr addr);

    /**
     * Génère le code pour l'initialisation d'un champ d'objet.
     * @param compiler
     * @param index Index du champ dans l'objet (offset dans le tas)
     */
    protected abstract void codeGenFieldARM(DecacCompiler compiler, int index);
}

    
