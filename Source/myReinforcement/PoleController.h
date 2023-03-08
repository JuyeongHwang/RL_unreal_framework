// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Components/InputComponent.h"
#include "PoleController.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class MYREINFORCEMENT_API UPoleController : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UPoleController();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

	UInputComponent* InputComponent;
	float CurrMotorSpeed;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	UPROPERTY(EditAnywhere)
		USceneComponent* BaseComponent;
	UPROPERTY(EditAnywhere)
		float MaxSpeed;
	UPROPERTY(EditAnywhere)
		float XBoundary;
	UPROPERTY(EditAnywhere)
		float MotorPower;

	void Move_XAxis(float AxisValue);
};
