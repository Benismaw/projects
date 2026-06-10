//classe pour les champs (Fields)
package fr.ensimag.deca.tree;

import java.io.PrintStream;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.ImmediateFloat;
import fr.ensimag.ima.pseudocode.ImmediateInteger;
import fr.ensimag.ima.pseudocode.NullOperand;
import fr.ensimag.deca.context.*;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.RegisterOffset;
import fr.ensimag.ima.pseudocode.instructions.*;;
public class DeclField extends AbstractDeclField{

    
    private Visibility visibility;
    private AbstractIdentifier typeField;
    private AbstractIdentifier nomField;
    private AbstractInitialization initialization;

    public DeclField(Visibility visibility,AbstractIdentifier typeField,AbstractIdentifier nomField,AbstractInitialization initialization){
        this.visibility=visibility;
        this.typeField=typeField;
        this.nomField=nomField;
        this.initialization=initialization;
    }
    //getters
    public Visibility geVisibility(){
        return this.visibility;
    }

    public AbstractIdentifier getTypeField(){
        return this.typeField;

    }
    public AbstractIdentifier getNomField(){
        return this.nomField;
        
    }

    public AbstractInitialization getInitialization(){
        return this.initialization;
        
    }
    protected void verifyFieldMembers(DecacCompiler compiler, 
            ClassDefinition currentClass,int index) throws ContextualError{
        // (verifier le type (pas void) et déclarer le champs
        Type typeField=this.typeField.verifyType(compiler);
        if (typeField.isVoid()){
            throw new ContextualError("Un champ ne peut pas être de type 'void' ", getLocation());
        }
        ClassDefinition superClass = currentClass.getSuperClass();
        if (superClass != null) {
            // on regarde dans l'environnement des membres de la super-classe
            ExpDefinition superDef = superClass.getMembers().get(this.nomField.getName());
            
            // si le nom est défini dans la super-classe, ce DOIT être un champ 
            if (superDef != null && !superDef.isField()) {
                throw new ContextualError("Le champ '" + nomField.getName() + 
                    "' ne peut pas redéfinir une méthode de la super-classe (Règle 2.5)", 
                    getLocation());
            }
        }
        // on cherche si un membre de même nom existe dans la super-classe
        FieldDefinition fieldDef = new FieldDefinition(typeField, getLocation(), this.visibility, currentClass, index);
        try {
            currentClass.getMembers().declare(this.nomField.getName(), fieldDef);
        } catch (EnvironmentExp.DoubleDefException e) {
            throw new ContextualError("Le champ '" + nomField.getName() + "' est déjà défini dans cette classe", getLocation());
        }

    //Décoration de l'identificateur
    this.nomField.setDefinition(fieldDef);
    
    }


    protected  void verifyFieldBody(DecacCompiler compiler, 
            ClassDefinition currentClass) throws ContextualError{
        // On recupere le type du champ
        Type typeField=this.typeField.verifyType(compiler);

        // On cree un environnement vide
        EnvironmentExp localEnv = new EnvironmentExp(null);

        // On verifie l'initialisation
        this.initialization.verifyInitialization(compiler,typeField,localEnv, currentClass);
            }

    // Initialisation par défaut (à 0, 0.0 ou null)
    protected void codeGenDefaultInit(DecacCompiler compiler){
        int index=nomField.getFieldDefinition().getIndex();
        Type type = nomField.getFieldDefinition().getType();
        if (type.isFloat()) {
            compiler.addInstruction(new LOAD(new ImmediateFloat(0.0f), Register.R0));
        } else if (type.isClass() || type.isNull()) {
            compiler.addInstruction(new LOAD(new NullOperand(), Register.R0));
        } else {
            // Pour int et boolean
            compiler.addInstruction(new LOAD(new ImmediateInteger(0), Register.R0));
        }
        // R1 contient déja l'adresse de l'objet this chargé dans DeclClass
        compiler.addInstruction(new STORE(Register.R0,new RegisterOffset(index,Register.R1)));

    }

    // Initialisation explicite (si une valeur est fournie)
    protected void codeGenExplicitInit(DecacCompiler compiler) {
        // à l'objet Initialization de faire  le soin de générer le STORE
        initialization.codeGenField(compiler, nomField.getFieldDefinition().getIndex());
    }

    @Override
    public void decompile(IndentPrintStream s){
        if (visibility==Visibility.PROTECTED){
            s.print("protected ");
        }
        typeField.decompile(s);
        s.print(" ");
        nomField.decompile(s);
        initialization.decompile(s);
        s.print(";");

    }
    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        typeField.prettyPrint(s, prefix, false);
        nomField.prettyPrint(s, prefix, false);
        initialization.prettyPrint(s, prefix, true);
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        typeField.iter(f);
        nomField.iter(f);
        initialization.iter(f);
    }
}
